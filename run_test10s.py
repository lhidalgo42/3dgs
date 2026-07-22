"""Validación con los primeros 10 segundos de IMG_0884 (300 frames, TODOS).

Área pequeña + cobertura densa = techo de calidad alcanzable. Si esto entrena
nítido, el problema del test anterior era densidad de vistas, y la corrida
grande con ~2500 frames va a funcionar.
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval, triangulation
from hloc.reconstruction import create_empty_db, get_image_ids, import_images
from hloc.triangulation import (estimation_and_geometric_verification,
                                import_features, import_matches)

ROOT = Path.home() / "dko-3dgs"
PY = str(ROOT / "gaussian-splatting" / "env" / "bin" / "python")
GLOMAP = str(Path.home() / "glomap-env" / "bin" / "glomap")

N_FRAMES = 300      # primeros 10 s a 30 fps
OVERLAP = 15
RETRIEVAL_K = 20
ITERATIONS = 30000

cand = ROOT / "candidates884"
test = ROOT / "hloc_10s"
timg = ROOT / "data_10s" / "input"
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


# 1. Primeros 300 frames (todos, sin selección)
stamp("1/6 primeros 300 frames")
frames = sorted(cand.glob("*.jpg"))[:N_FRAMES]
assert len(frames) == N_FRAMES, f"solo hay {len(frames)}"
if timg.exists():
    shutil.rmtree(timg)
timg.mkdir(parents=True)
for f in frames:
    os.link(f, timg / f.name)
names = sorted(p.name for p in timg.glob("*.jpg"))

if test.exists():
    shutil.rmtree(test)
test.mkdir()

# 2. Features + pares
stamp("2/6 features + pares")
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

# 3. Matching + db + verificación
stamp("3/6 matching + db")
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

# 4. Mapper incremental (ganador en este video)
stamp("4/6 mapper incremental")
inc_out = test / "incremental"
inc_out.mkdir()
recs = pycolmap.incremental_mapping(db, timg, inc_out)
models = [(inc_out / str(k), v) for k, v in recs.items()]
assert models, "sin modelo"
sfm_path, best = max(models, key=lambda t: t[1].num_reg_images())
print(f"incremental: {best.num_reg_images()}/{len(names)} registradas, "
      f"{best.num_points3D()} pts, err {best.compute_mean_reprojection_error():.2f}px",
      flush=True)

# 5. Retriangulación + BA
stamp("5/6 retriangulación + BA")
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

# 6. Undistort + entrenamiento con densificación agresiva
stamp("6/6 undistort + entrenamiento")
data = ROOT / "data_10s"
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

MODEL_DIR = ROOT / "output" / "dko3d_test10s"
r = subprocess.run(
    [PY, "train.py", "-s", str(data), "-m", str(MODEL_DIR),
     "-r", "2", "--data_device", "cpu",
     "--iterations", str(ITERATIONS),
     "--save_iterations", str(ITERATIONS), "--test_iterations", "-1",
     "--densify_grad_threshold", "0.0001",
     "--densify_until_iter", "20000"],
    cwd=ROOT / "gaussian-splatting")
if r.returncode:
    sys.exit(f"train.py falló con código {r.returncode}")
stamp(f"PIPELINE_DONE modelo en {MODEL_DIR}")
