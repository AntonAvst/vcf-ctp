#!/usr/bin/env python3
"""
train_pose.py — Train a YOLOv8-Pose model on the new 26-keypoint cow dataset.

This is a standalone training script. It is NOT wired into vcf-ctp's
config.py / drive_manager.py — training happens locally against a
Roboflow-exported dataset directory, same pattern the project already
used for the cow_seg model ("trained locally via Ultralytics, not
Roboflow-hosted training").

Usage
-----
    pip install ultralytics pyyaml

    # Fresh training run, starting from Ultralytics' COCO-pretrained
    # pose checkpoint (backbone transfer only — the pose head is
    # reinitialized because 17 kp != 26 kp):
    python3 train_pose.py --data /path/to/cow-pose-estimation-2/data.yaml \
        --model yolov8n-pose.pt --epochs 300 --imgsz 640 --batch 16

    # Starting from the project's existing 19-kp cow_pose/best.pt instead
    # (also backbone-only transfer, same reason):
    python3 train_pose.py --data /path/to/data.yaml \
        --model ~/thesis_workspace/vcf-ctp/data/models/cow_pose/best.pt \
        --epochs 300

    # Validate an already-trained model against the dataset, no training:
    python3 train_pose.py --data /path/to/data.yaml --model runs/pose/cow_pose_26kp/weights/best.pt --val_only

Notes
-----
- Ultralytics builds the pose head's output shape from `kpt_shape` in
  data.yaml, not from whatever the starting checkpoint had. If `--model`
  points at a .pt file whose kpt_shape differs from this dataset's (26),
  Ultralytics transfers every matching backbone/neck tensor and skips
  (reinitializes) the ones that don't match shape — you'll see a
  "transferred N/M items from pretrained weights" line confirming this.
  That's expected and is the normal way to retarget a pose model to a
  new keypoint set.
- `flip_idx` in data.yaml must have exactly 26 entries (one per keypoint,
  itself if symmetric, its mirror's index if it's a left/right pair) or
  left-right augmentation will silently corrupt keypoints during training.
  This script just warns if the length doesn't match; it doesn't try to
  guess a fix.
- Roboflow's data.yaml paths (`../train/images` etc.) are relative to the
  folder data.yaml itself sits in. If you moved data.yaml, fix those paths
  or pass an absolute-path copy.
"""
import argparse
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8-Pose on a custom keypoint dataset")
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--model", default="yolov8n-pose.pt",
                   help="starting weights (.pt) or architecture (.yaml). Default: Ultralytics' nano "
                        "COCO-pose checkpoint (yolov8n-pose.pt) — a generic, publicly-distributed "
                        "checkpoint, NOT your project's existing cow_pose/best.pt. This script never "
                        "reads or writes your old model unless you explicitly pass its path here.")
    p.add_argument("--scratch", action="store_true",
                   help="ignore --model entirely and build a brand-new model from the "
                        "yolov8n-pose.yaml architecture with randomly-initialized weights — "
                        "no pretrained weights of any kind (not even generic COCO ones). "
                        "Needs more epochs/data to converge than transfer learning.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="'0', '0,1', or 'cpu'")
    p.add_argument("--patience", type=int, default=50, help="early-stop patience, epochs with no val improvement")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="runs/pose", help="where Ultralytics writes run folders")
    p.add_argument("--name", default="cow_pose_26kp", help="run folder name under --project")
    p.add_argument("--resume", action="store_true", help="resume the last checkpoint in --project/--name")
    p.add_argument("--pose_loss_gain", type=float, default=None,
                   help="override the pose-loss weight (Ultralytics default is 12.0); "
                        "custom keypoint sets sometimes train better lower, e.g. 5.0")
    p.add_argument("--val_only", action="store_true", help="skip training; just run model.val() on --data")
    return p.parse_args()


def validate_data_yaml(data_path: Path) -> tuple[Path, dict]:
    if not data_path.exists():
        sys.exit(f"ERROR: {data_path} does not exist")

    if data_path.is_dir():
        candidates = sorted(data_path.glob("*.yaml")) + sorted(data_path.glob("*.yml"))
        candidates = [c for c in candidates if "Zone.Identifier" not in c.name]
        if not candidates:
            sys.exit(f"ERROR: {data_path} is a directory with no data.yaml/.yml inside it")
        if len(candidates) > 1:
            sys.exit(f"ERROR: {data_path} contains multiple yaml files ({[c.name for c in candidates]}); "
                      f"pass the exact file via --data instead of the directory")
        data_path = candidates[0]
        print(f"--data was a directory; using {data_path}")

    with open(data_path) as f:
        cfg = yaml.safe_load(f)

    kpt_shape = cfg.get("kpt_shape")
    if not kpt_shape or len(kpt_shape) != 2:
        sys.exit(f"ERROR: {data_path} has no valid 'kpt_shape: [n_kp, dims]' entry")
    n_kp, dims = kpt_shape
    print(f"Dataset: {n_kp} keypoints, {dims} dims/keypoint, class(es): {cfg.get('names')}")

    flip_idx = cfg.get("flip_idx", [])
    if len(flip_idx) != n_kp:
        print(f"WARNING: flip_idx has {len(flip_idx)} entries, expected {n_kp}. "
              f"Horizontal-flip augmentation will mis-map keypoints until this is fixed.")

    base = data_path.parent
    for split_key in ("train", "val", "test"):
        rel = cfg.get(split_key)
        if not rel:
            continue
        resolved = (base / rel).resolve()
        print(f"  {split_key:5s} -> {resolved}  [{'OK' if resolved.exists() else 'MISSING'}]")

    return data_path, cfg


def main():
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    data_path, cfg = validate_data_yaml(data_path)

    # Imported after the cheap yaml check above, so a bad --data path fails
    # fast without paying Ultralytics' import cost.
    from ultralytics import YOLO

    if args.resume:
        # Resuming means continuing THIS run's own in-progress checkpoint, not
        # --model (which is only the starting point for a fresh run). Ultralytics
        # needs the actual last.pt — it carries the saved optimizer state, epoch
        # count, etc. that --resume relies on.
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_pt.exists():
            sys.exit(f"ERROR: --resume was passed but {last_pt} doesn't exist — "
                      f"nothing to resume. Check --project/--name match the original run.")
        model_source = str(last_pt)
        print(f"Weights source: {model_source}  (resuming in-progress run)")
    elif args.scratch:
        print("Weights source: NONE — building yolov8n-pose.yaml architecture from random init "
              "(--scratch was passed; --model is ignored).")
        model_source = "yolov8n-pose.yaml"
    else:
        model_source = args.model
        print(f"Weights source: {model_source}")
        if "cow_pose" in str(model_source) or "best.pt" in str(model_source):
            print(f"  -> NOTE: this path looks like it could be your project's existing cow_pose "
                  f"model. It will be READ for backbone transfer only — this script never writes "
                  f"back to it. New weights always land under --project/--name "
                  f"({args.project}/{args.name}/weights/).")

    model = YOLO(model_source)

    if args.val_only:
        metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)
        print(metrics)
        return

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=True,
    )
    if args.pose_loss_gain is not None:
        train_kwargs["pose"] = args.pose_loss_gain

    results = model.train(**train_kwargs)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining finished.")
    print(f"Best weights: {best}")
    print(f"\nNext steps:")
    print(f"  1. python3 train_pose.py --data {data_path} --model {best} --val_only")
    print(f"  2. Spot-check predictions on a few held-out crops before trusting mAP alone — "
          f"OKS-based pose mAP can look fine while specific landmarks (e.g. a tail tip that's "
          f"rarely visible) are consistently off.")
    print(f"  3. Copy {best} into the vcf-ctp data tree as models/cow_pose/best.pt "
          f"(or a new versioned filename) and update params.yaml's models.pose path.")
    print(f"  4. Do NOT point track_and_dump.py at this model yet — see the integration "
          f"checklist; schema.py and the array widths in track_and_dump.py still assume 19 kp.")


if __name__ == "__main__":
    main()