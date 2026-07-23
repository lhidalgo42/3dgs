"""Test del video Blackmagic A001 (54s, 4K60, SDR, sin estabilización).

Extrae frames -> selección nítida (1 de 6, ~540) -> seq+NetVLAD -> matching ->
mapper incremental -> retriangulación + BA -> undistort. SE DETIENE antes de
entrenar (SFM_DONE) para no pelear la GPU con el entrenamiento del test 10s;
el entrenamiento se lanza aparte cuando la GPU se libere.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval, triangulation
from hloc.reconstruction import create_empty_db, get_image_ids, import_images
from hloc.triangulation import (estimation_and_geometric_verification,
                                import_features, import_matches)

ROOT = Path.home() / "dko-3dgs"
VIDEO = ROOT / "A001_07221626_C001.mov"
WIN = 6          # 60 fps -> ~10 nítidos/seg de video
OVERLAP = 15
RETRIEVAL_K = 20

cand = ROOT / "candidatesA001"
test = ROOT / "hloc_A001"
timg = ROOT / "data_A001" / "input"
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


stamp("1/6 extracción de frames")
if not cand.exists() or len(list(cand.glob("*.jpg"))) < 3000:
    if cand.exists():
        shutil.rmtree(cand)
    cand.mkdir()
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", str(VIDEO), "-qscale:v", "2", str(cand / "%05d.jpg")])
    if r.returncode:
        sys.exit("ffmpeg falló")
frames = sorted(cand.glob("*.jpg"))
print(f"{len(frames)} frames", flush=True)

stamp("2/6 selección de nítidos")
if timg.exists():
    shutil.rmtree(timg)
timg.mkdir(parents=True)
for i in range(0, len(frames), WIN):
    window = frames[i:i + WIN]
    best = max(window, key=lambda f: cv2.Laplacian(
        cv2.imread(str(f), cv2.IMREAD_REDUCED_GRAYSCALE_4), cv2.CV_64F).var())
    os.link(best, timg / best.name)
names = sorted(p.name for p in timg.glob("*.jpg"))
print(f"{len(names)} seleccionados (ventana {WIN})", flush=True)

if test.exists():
    shutil.rmtree(test)
test.mkdir()

stamp("3/6 features + pares")
feats_h5 = extract_features.main(extract_features.confs["aliked-n16"], timg, test)
global_desc = extract_features.main(extract_features.confs["netvlad"], timg, test)
pairs_all = set()
for i, a in enumerate(names):
    for j in range(i + 1, min(i + 1 + OVERLAP, len(names))):
        pairs_all.add((a, names[j]))
pairs_retr = test / "pairs_retrieval.txt"
pairs_from_retrieval.main(global_desc, pairs_retr, num_matched=RETRIEVAL_K)
for line in pairs_retr.read_text().splitlines():
    a, b = line.split()
    if a != b and (b, a) not in pairs_all:
        pairs_all.add((a, b))
pairs_path = test / "pairs.txt"
pairs_path.write_text("\n".join(f"{a} {b}" for a, b in sorted(pairs_all)) + "\n")
print(f"{len(pairs_all)} pares", flush=True)

stamp("4/6 matching + db + verificación")
matches_h5 = test / "matches.h5"
match_features.main(match_features.confs["aliked+lightglue"], pairs_path, feats_h5,
                    matches=matches_h5)
sfm_dir = test / "sfm"
sfm_dir.mkdir()
db = sfm_dir / "database.db"
create_empty_db(db)
import_images(timg, db, pycolmap.CameraMode.SINGLE, options=dict(camera_model="OPENCV"))
image_ids = get_image_ids(db)
with pycolmap.Database.open(db) as dbh:
    import_features(image_ids, dbh, feats_h5)
    import_matches(image_ids, dbh, pairs_path, matches_h5,
                   min_match_score=None, skip_geometric_verification=False)
estimation_and_geometric_verification(db, pairs_path)

stamp("5/6 mapper incremental")
inc_out = test / "incremental"
inc_out.mkdir()
recs = pycolmap.incremental_mapping(db, timg, inc_out)
models = [(inc_out / str(k), v) for k, v in recs.items()]
assert models, "sin modelo"
sfm_path, best = max(models, key=lambda t: t[1].num_reg_images())
print(f"incremental: {best.num_reg_images()}/{len(names)} registradas, "
      f"{best.num_points3D()} pts, err {best.compute_mean_reprojection_error():.2f}px",
      flush=True)

stamp("6/6 retriangulación + BA + undistort")
reg_names = {best.image(i).name for i in best.reg_image_ids()}
pairs_tri = test / "pairs_tri.txt"
pairs_tri.write_text("\n".join(
    f"{a} {b}" for a, b in sorted(pairs_all) if a in reg_names and b in reg_names) + "\n")
sfm_tri = test / "tri"
rec = triangulation.main(sfm_tri, sfm_path, timg, pairs_tri, feats_h5, matches_h5)
print(f"retriangulado: {rec.num_points3D()} puntos", flush=True)
pycolmap.bundle_adjustment(rec, pycolmap.BundleAdjustmentOptions())
rec.write(str(sfm_tri))
print(f"BA: err {rec.compute_mean_reprojection_error():.3f}px", flush=True)

data = ROOT / "data_A001"
for d in (data / "images", data / "sparse", data / "stereo", data / "distorted"):
    if d.exists():
        shutil.rmtree(d)
pycolmap.undistort_images(output_path=str(data), input_path=str(sfm_tri),
                          image_path=str(timg), output_type="COLMAP")
sparse0 = data / "sparse" / "0"
sparse0.mkdir(parents=True, exist_ok=True)
for f in (data / "sparse").iterdir():
    if f.is_file():
        shutil.move(str(f), sparse0 / f.name)

stamp("SFM_DONE — listo para entrenar (data_A001)")
