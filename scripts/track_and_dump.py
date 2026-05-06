#!/usr/bin/env python3
"""
track_and_dump.py — detector + tracker (+ optional pose + appearance embedding)

- Loads an Ultralytics YOLO detector (your cow detector best.pt)
- Tracks with ByteTrack or BoT-SORT
- Writes tracks.csv and tracks.jsonl
- (Optional) Saves crops and 128D embeddings
- (Optional) Runs a YOLO-Pose model on detection crops and writes keypoints
  (currently a 19-keypoint cow layout)

WSL-friendly, no GUI required.
"""

import argparse, csv, json, os
from pathlib import Path
from time import time

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# -------------------- Embedding backbone (MobileNetV3-Small -> 128D) --------------------
import torch
import torch.nn as nn
import torchvision.models as tv
from collections import defaultdict  # per-ID crop counting

EXPECTED_KP = 19  # new model: 19 keypoints (nose, forehead, ..., udder_center, neck)


def log(msg: str) -> None:
    print(f"[track] {msg}", flush=True)


class Embedder128(nn.Module):
    def __init__(self, pretrained=True, out_dim=128):
        super().__init__()
        m = tv.mobilenet_v3_small(
            weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = m.features  # (B,C,H,W)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(576, out_dim)  # mobilenet_v3_small last channels = 576

    def forward(self, x):  # x: (B,3,224,224)
        f = self.backbone(x)  # (B,576,h,w)
        f = self.pool(f).flatten(1)  # (B,576)
        z = self.proj(f)  # (B,128)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return z


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Detector .pt")
    ap.add_argument("--source", required=True, help="Video path")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--camera_id", default="cam0")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save_crops", action="store_true")
    ap.add_argument("--save_embed", action="store_true")

    # control crop saving cadence per ID
    ap.add_argument(
        "--crop_every",
        type=int,
        default=1,
        help="Save crop every N-th appearance per track ID (1 = every frame for that ID).",
    )

    # Embedding configs
    ap.add_argument("--embed_size", type=int, default=128)
    ap.add_argument("--embed_batch", type=int, default=64)

    # Pose options
    ap.add_argument("--pose_model", default="", help="Ultralytics YOLO-Pose .pt (optional)")
    ap.add_argument("--pose_imgsz", type=int, default=384)
    ap.add_argument("--pose_conf", type=float, default=0.25)
    ap.add_argument("--pose_batch", type=int, default=32)
    ap.add_argument("--write_kp_jsonl", action="store_true", help="Write keypoints.jsonl")

    # Pose keypoint confidence -> visibility flag mapping
    # We will encode v as:
    #   2 = confident (>= pose_kp_conf_thresh)
    #   1 = low confidence but present (< thresh)
    #   0 = missing/ignored
    ap.add_argument(
        "--pose_kp_conf_thresh",
        type=float,
        default=0.30,
        help="Keypoint confidence threshold used to set v=2 (>=thr) vs v=1 (<thr).",
    )

    # Save crop only if (crop_w, crop_h) >= this threshold
    ap.add_argument(
        "--min_crop_wh",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(0, 0),
        help="Only save crops if the cropped bbox (after margin/clamp) is at least W x H pixels.",
    )

    return ap.parse_args()


# -------------------- utils --------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_tensor_bchw(img_bgr, size=224):
    """Resize to size x size, BGR->RGB, HWC->BCHW, float [0,1], normalize ImageNet."""
    x = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)  # CHW
    return x


def crops_from_bboxes(frame, bboxes, margin=0.1):
    """Return crops list and crop_boxes (x1,y1,x2,y2) actually used after margin & clamping."""
    H, W = frame.shape[:2]
    crops, used = [], []
    for (x1, y1, x2, y2) in bboxes:
        w = x2 - x1
        h = y2 - y1
        mx = int(round(w * margin))
        my = int(round(h * margin))
        xx1 = max(0, int(x1) - mx)
        yy1 = max(0, int(y1) - my)
        xx2 = min(W - 1, int(x2) + mx)
        yy2 = min(H - 1, int(y2) + my)
        crops.append(frame[yy1:yy2, xx1:xx2].copy())
        used.append((xx1, yy1, xx2, yy2))
    return crops, used


def flat_kps_xyv(kps_xyv):
    """kps_xyv: (K,3) -> flat list [x1,y1,v1,...] floats (rounded for CSV-size)."""
    out = []
    for (x, y, v) in kps_xyv:
        out.extend([float(round(x, 3)), float(round(y, 3)), int(v)])
    return out


# -------------------- main --------------------
def main():
    args = parse_args()

    outdir = ensure_dir(Path(args.outdir))
    crops_dir = ensure_dir(outdir / "crops") if args.save_crops else None

    # open outputs
    csv_path = outdir / "tracks.csv"
    jsonl_path = outdir / "tracks.jsonl"
    kp_jsonl_path = outdir / "keypoints.jsonl" if args.write_kp_jsonl else None

    log(f"Starting tracking")
    log(f"  model     : {args.model}")
    log(f"  source    : {args.source}")
    log(f"  outdir    : {outdir}")
    log(f"  tracker   : {args.tracker}")
    log(f"  save_crops: {args.save_crops}, save_embed: {args.save_embed}")
    if args.pose_model:
        log(f"  pose_model: {args.pose_model}")

    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    header = [
        "camera_id",
        "frame_index",
        "frame_time_sec",
        "temp_id",
        "det_conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "w",
        "h",
    ]
    if args.save_embed:
        header.append("embed")  # JSON list of 128 floats
    # pose fields
    if args.pose_model:
        # kps/kps_norm are flat [x,y,v,...]
        # kps_conf is mean confidence (float)
        # kps_kconf is per-keypoint confidence list (JSON)
        header += ["kps", "kps_norm", "kps_conf", "kps_kconf"]
    csv_w.writerow(header)
    csv_f.flush()

    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    # models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    det_model = YOLO(args.model)
    det_model.fuse()

    # Embedding model
    embedder = None
    if args.save_embed:
        embedder = Embedder128(pretrained=True, out_dim=args.embed_size).to(device).eval()

    # Pose model
    pose_model = None
    if args.pose_model:
        pose_model = YOLO(args.pose_model)
        pose_model.fuse()

    # video stream
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    log(f"Video opened: {W}x{H} @ {fps:.2f} FPS, frames≈{total_frames if total_frames > 0 else 'unknown'}")
    log(f"Writing CSV to: {csv_path}")
    log(f"Writing JSONL to: {jsonl_path}")

    frame_idx = 0
    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=f"Tracking {Path(args.source).name}",
        unit="frame",
        dynamic_ncols=True,
    )

    # per-ID crop occurrence counter
    crop_occurrence = defaultdict(int)
    warned_kp_mismatch = False

    # for periodic flushing and simple ETA-ish info
    last_flush_frame = 0
    FLUSH_EVERY = 50  # frames

    t_start = time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # run detector+tracker on this single frame
        results = det_model.track(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
            persist=True,
            verbose=False,
        )
        if not results:
            frame_idx += 1
            pbar.update(1)
            # periodic flush / status
            if frame_idx - last_flush_frame >= FLUSH_EVERY:
                csv_f.flush()
                jsonl_f.flush()
                pbar.set_postfix(frames=frame_idx)
                last_flush_frame = frame_idx
            continue

        r = results[0]
        if r.boxes is None or r.boxes.xyxy is None:
            frame_idx += 1
            pbar.update(1)
            if frame_idx - last_flush_frame >= FLUSH_EVERY:
                csv_f.flush()
                jsonl_f.flush()
                pbar.set_postfix(frames=frame_idx)
                last_flush_frame = frame_idx
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = (
            r.boxes.conf.cpu().numpy()
            if r.boxes.conf is not None
            else np.zeros(len(xyxy), dtype=np.float32)
        )
        tids_raw = r.boxes.id
        if tids_raw is None:
            frame_idx += 1
            pbar.update(1)
            if frame_idx - last_flush_frame >= FLUSH_EVERY:
                csv_f.flush()
                jsonl_f.flush()
                pbar.set_postfix(frames=frame_idx)
                last_flush_frame = frame_idx
            continue
        tids = tids_raw.cpu().numpy().astype(int)

        # crops (for embedding and pose)
        crops, used_boxes = crops_from_bboxes(frame, xyxy, margin=0.10)

        # ------- embeddings (optional) -------
        embeds_json = [None] * len(crops)
        if args.save_embed and len(crops):
            batch = [to_tensor_bchw(c, size=224) for c in crops]
            X = torch.from_numpy(np.stack(batch)).to(device)
            with torch.no_grad():
                Z = embedder(X).cpu().numpy()  # (N,128)
            embeds_json = [
                json.dumps(list(map(lambda f: round(float(f), 4), z.tolist()))) for z in Z
            ]

        # ------- pose (optional) -------
        kps_list = [None] * len(crops)
        kpsn_list = [None] * len(crops)
        kpm_list = [None] * len(crops)  # mean
        kpc_list = [None] * len(crops)  # per-kp
        if pose_model and len(crops):
            pose_res = pose_model.predict(
                source=crops,
                imgsz=args.pose_imgsz,
                conf=args.pose_conf,
                verbose=False,
                stream=False,
            )
            for i, pres in enumerate(pose_res):
                if pres.keypoints is None or pres.keypoints.data is None or len(pres.keypoints.data) == 0:
                    continue
                k = pres.keypoints
                xy = k.xy[0].cpu().numpy()  # (K,2)
                sc = k.conf[0].cpu().numpy() if k.conf is not None else None  # (K,)
                K = xy.shape[0]

                # sanity check: expect 19 keypoints for the new model
                if EXPECTED_KP is not None and K != EXPECTED_KP and not warned_kp_mismatch:
                    log(
                        f"Warning: pose model returned {K} keypoints, "
                        f"but EXPECTED_KP={EXPECTED_KP}. Proceeding anyway."
                    )
                    warned_kp_mismatch = True

                # visibility flag v in {0,1,2} (Ultralytics convention in labels)
                # Here we map predicted keypoint confidence to v:
                #  - v=2 if conf >= pose_kp_conf_thresh
                #  - v=1 if 0 < conf < pose_kp_conf_thresh
                #  - v=0 if conf is missing
                if sc is not None:
                    thr = float(args.pose_kp_conf_thresh)
                    vis = np.where(sc >= thr, 2, 1).astype(int)
                else:
                    vis = np.ones(K, dtype=int) * 2

                (cx1, cy1, cx2, cy2) = used_boxes[i]
                crop_w = max(1, cx2 - cx1)
                crop_h = max(1, cy2 - cy1)

                # pres.orig_img is the crop as fed to the model; resize ratios usually = 1
                ow = pres.orig_img.shape[1]
                oh = pres.orig_img.shape[0]
                sx = crop_w / max(1, ow)
                sy = crop_h / max(1, oh)

                xy_full = np.zeros((K, 2), dtype=np.float32)
                xy_full[:, 0] = cx1 + xy[:, 0] * sx
                xy_full[:, 1] = cy1 + xy[:, 1] * sy

                xy_norm = np.zeros_like(xy_full)
                xy_norm[:, 0] = xy_full[:, 0] / W
                xy_norm[:, 1] = xy_full[:, 1] / H

                kps_xyv = np.concatenate([xy_full, vis.reshape(-1, 1)], axis=1)
                kpsn_xyv = np.concatenate([xy_norm, vis.reshape(-1, 1)], axis=1)

                kps_list[i] = json.dumps(flat_kps_xyv(kps_xyv))
                kpsn_list[i] = json.dumps(flat_kps_xyv(kpsn_xyv))
                kpm_list[i] = float(np.nanmean(sc)) if sc is not None else 0.0
                kpc_list[i] = json.dumps([
                    float(round(c, 4)) for c in sc.tolist()
                ]) if sc is not None else "[]"

        # write rows + optionally save crops
        t_sec = frame_idx / max(1e-6, fps)
        for j, (box, tid, conf) in enumerate(zip(xyxy, tids, confs)):
            x1, y1, x2, y2 = box.tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = (x2 - x1)
            h = (y2 - y1)

            row = [
                args.camera_id,
                frame_idx,
                round(t_sec, 3),
                int(tid),
                float(conf),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(cx),
                float(cy),
                float(w),
                float(h),
            ]
            if args.save_embed:
                row.append(embeds_json[j] if embeds_json[j] is not None else "[]")
            if pose_model:
                row.append(kps_list[j] if kps_list[j] is not None else "[]")
                row.append(kpsn_list[j] if kpsn_list[j] is not None else "[]")
                row.append(kpm_list[j] if kpm_list[j] is not None else 0.0)
                row.append(kpc_list[j] if kpc_list[j] is not None else "[]")

            csv_w.writerow(row)

            obj = {
                "camera_id": args.camera_id,
                "frame_index": frame_idx,
                "frame_time_sec": round(t_sec, 3),
                "temp_id": int(tid),
                "det_conf": float(conf),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
            }
            if args.save_embed and embeds_json[j] is not None:
                obj["embed"] = json.loads(embeds_json[j])
            if pose_model and kps_list[j] is not None:
                obj["kps"] = json.loads(kps_list[j])
                obj["kps_norm"] = json.loads(kpsn_list[j])
                obj["kps_conf"] = kpm_list[j]
                obj["kps_kconf"] = json.loads(kpc_list[j]) if kpc_list[j] is not None else []
            jsonl_f.write(json.dumps(obj) + "\n")

            # optional crop dump (every N-th occurrence per ID) + minimum crop size
            if crops_dir is not None:
                tid_i = int(tid)
                crop_occurrence[tid_i] += 1

                if crop_occurrence[tid_i] % max(1, args.crop_every) == 0:
                    (cx1, cy1, cx2, cy2) = used_boxes[j]
                    crop_w = int(cx2 - cx1)
                    crop_h = int(cy2 - cy1)

                    min_w, min_h = args.min_crop_wh
                    if crop_w >= int(min_w) and crop_h >= int(min_h):
                        crop = frame[cy1:cy2, cx1:cx2]
                        out_name = f"{args.camera_id}_id{tid_i:04d}_f{frame_idx:06d}.jpg"
                        cv2.imwrite(str(crops_dir / out_name), crop)


        frame_idx += 1
        pbar.update(1)

        # periodic flush + status update
        if frame_idx - last_flush_frame >= FLUSH_EVERY:
            csv_f.flush()
            jsonl_f.flush()
            elapsed = time() - t_start
            fps_proc = frame_idx / max(elapsed, 1e-6)
            pbar.set_postfix(frames=frame_idx, fps=f"{fps_proc:.1f}")
            last_flush_frame = frame_idx

    pbar.close()
    csv_f.flush()
    jsonl_f.flush()
    csv_f.close()
    jsonl_f.close()
    cap.release()

    t_total = time() - t_start
    log(f"Done. Processed {frame_idx} frames in {t_total:.1f}s ({frame_idx / max(t_total,1e-6):.1f} FPS).")
    log(f"tracks.csv : {csv_path}")
    log(f"tracks.jsonl: {jsonl_path}")
    if crops_dir is not None:
        log(f"crops saved to: {crops_dir}")
    if kp_jsonl_path:
        # currently unused – you can later stream only keypoints here if needed
        pass


if __name__ == "__main__":
    main()


# Example:
# source ~/venvs/cowtrack/bin/activate
# cd /home/anton/thesis_workspace/vcf-ctp/scripts

# DET=/home/anton/thesis_workspace/vcf-ctp/models/cow_detector/best.pt
# POSE=/home/anton/thesis_workspace/vcf-ctp/models/cow_pose/best.pt
# VID=/home/anton/thesis_workspace/raw_data/calving/6558/refet_33_S20241221070000_E20241221080000_6558.mp4
# OUT=/home/anton/thesis_workspace/outputs/tracks/refet33_2024-12-21_pose

# python3 track_and_dump.py \
#   --model "$DET" \
#   --source "$VID" \
#   --outdir "$OUT" \
#   --tracker "/home/anton/thesis_workspace/vcf-ctp/configs/botsort.yaml" \
#   --camera_id "refet_33" \
#   --imgsz 960 --conf 0.30 --iou 0.60 \
#   --save_crops --save_embed \
#   --pose_model "$POSE" --pose_imgsz 384 --pose_conf 0.25 --pose_batch 32 \
#   --crop_every 100 \
#   --min_crop_wh 100 100
