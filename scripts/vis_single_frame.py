import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# ----- paths -----
ANN_PATH = Path("_annotations.coco.json")  # adjust if needed
IMG_DIR = ANN_PATH.parent
IMG_NAME = "refet_33_id0011_f006429_jpg.rf.53dbbb90cd37d596a15612259117537a.jpg"

# ----- load annotations -----
with open(ANN_PATH, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
annos_by_img = {}
for ann in coco["annotations"]:
    annos_by_img.setdefault(ann["image_id"], []).append(ann)

cats = {c["id"]: c for c in coco["categories"]}
kp_names = cats[1]["keypoints"]          # 19 keypoints, last is "neck"
skel_pairs = cats[1]["skeleton"]         # 1-based indices

# convert skeleton to 0-based (for Python indexing)
skeleton = [(a - 1, b - 1) for a, b in skel_pairs]

# ----- find the image entry -----
img_id = None
for iid, img in images.items():
    if img["file_name"] == IMG_NAME:
        img_id = iid
        break

if img_id is None:
    raise RuntimeError(f"Image {IMG_NAME} not found in COCO file")

img_info = images[img_id]
h, w = img_info["height"], img_info["width"]
img_path = IMG_DIR / img_info["file_name"]

# ----- load image -----
img_bgr = cv2.imread(str(img_path))
if img_bgr is None:
    raise RuntimeError(f"Failed to read image: {img_path}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ----- draw function -----
def draw_keypoints_and_skeleton(img, keypoints, color=(0, 255, 0), radius=3, thickness=2):
    """
    keypoints: flat list [x1, y1, v1, ..., xK, yK, vK] in COCO format.
    v: 0=unlabeled, 1=visible, 2=occluded
    """
    K = len(keypoints) // 3
    pts = []
    for i in range(K):
        x = keypoints[3 * i + 0]
        y = keypoints[3 * i + 1]
        v = keypoints[3 * i + 2]
        pts.append((x, y, v))

    # draw points
    for idx, (x, y, v) in enumerate(pts):
        if v > 0:  # draw both visible and occluded
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            cv2.putText(
                img,
                str(idx),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # draw skeleton lines only if both endpoints are labeled (v>0)
    for a, b in skeleton:
        if a < 0 or b < 0 or a >= K or b >= K:
            continue
        x1, y1, v1 = pts[a]
        x2, y2, v2 = pts[b]
        if v1 > 0 and v2 > 0:
            cv2.line(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

# ----- pick the first annotation for this image (there is one cow) -----
ann_list = annos_by_img.get(img_id, [])
if not ann_list:
    raise RuntimeError(f"No annotations for image id={img_id}")

ann = ann_list[0]
kps = ann["keypoints"]

vis_img = img_rgb.copy()
draw_keypoints_and_skeleton(vis_img, kps, color=(255, 0, 0))

plt.figure(figsize=(4, 6))
plt.imshow(vis_img)
plt.axis("off")
plt.title(IMG_NAME)

out_path = "vis_output.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved visualization to {out_path}")

