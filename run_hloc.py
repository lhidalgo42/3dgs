"""SfM with SuperPoint + LightGlue (GPU) + pycolmap mapper, sequential pairs."""
from pathlib import Path
import pycolmap
from hloc import extract_features, match_features, reconstruction

root = Path.home() / "dko-3dgs"
images = root / "data/input"
out = root / "hloc_out"
out.mkdir(exist_ok=True)

names = sorted(p.name for p in images.glob("*.jpg"))
assert names, "no images"
OVERLAP = 15
pairs = out / "pairs.txt"
with open(pairs, "w") as f:
    for i, a in enumerate(names):
        for j in range(i + 1, min(i + 1 + OVERLAP, len(names))):
            f.write(f"{a} {names[j]}\n")
print(f"{len(names)} images, sequential pairs written")

feat_conf = extract_features.confs["aliked-n16"]
match_conf = match_features.confs["aliked+lightglue"]
features = extract_features.main(feat_conf, images, out)
matches = match_features.main(match_conf, pairs, feat_conf["output"], out)

model = reconstruction.main(
    out / "sfm", images, pairs, features, matches,
    camera_mode=pycolmap.CameraMode.SINGLE,
    image_options=dict(camera_model="OPENCV"),
)
print("REGISTERED:", model.num_reg_images() if model else 0)
