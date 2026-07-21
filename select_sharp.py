"""Pick the sharpest frame (Laplacian variance) from each window of N candidates."""
import cv2, sys, shutil
from pathlib import Path

src, dst, win = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
frames = sorted(src.glob("*.jpg"))
assert frames, f"no frames in {src}"

kept = 0
for i in range(0, len(frames), win):
    window = frames[i:i + win]
    best = max(window, key=lambda f: cv2.Laplacian(
        cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var())
    shutil.copy(best, dst / f"{kept:04d}.jpg")
    kept += 1
print(f"kept {kept} of {len(frames)}")
