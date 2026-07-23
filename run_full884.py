"""Corrida DEFINITIVA de IMG_0884 completo con la receta validada.

Selección nítida 1-de-10 (~2700) -> seq+NetVLAD -> matching -> mapper
incremental -> retriangulación + BA -> subset de 2000 cámaras para entrenar
(RAM) con nube de puntos completa -> 3DGS a -r 4 (validado en test 10s).
Salida: output/dko3d_884full
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
PY = str(ROOT / "gaussian-splatting" / "env" / "bin" / "python")

WIN = 10             # 26754 frames -> ~2675 nítidos (1 cada 0.33 s)
OVERLAP = 15
RETRIEVAL_K = 20
MAX_TRAIN = 2000
ITERATIONS = 30000
MODEL_DIR = ROOT / "output" / "dko3d_884full"

cand = ROOT / "candidates884"
test = ROOT / "hloc_884full"
timg = ROOT / "data_884full" / "input"
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


# 1. Selección nítida
stamp("1/7 selección de nítidos (1 de 10)")
frames = sorted(cand.glob("*.jpg"))
assert len(frames) > 26000, "faltan frames en candidates884"
if timg.exists():
    shutil.rmtree(timg)
timg.mkdir(parents=True)
for i in range(0, len(frames), WIN):
    window = frames[i:i + WIN]
    best = max(window, key=lambda f: cv2.Laplacian(
        cv2.imread(str(f), cv2.IMREAD_REDUCED_GRAYSCALE_4), cv2.CV_64F).var())
    os.link(best, timg / best.name)
names = sorted(p.name for p in timg.glob("*.jpg"))
print(f"{len(names)} seleccionados", flush=True)

if test.exists():
    shutil.rmtree(test)
test.mkdir()

# 2. Features + pares
stamp("2/7 features ALIKED + NetVLAD + pares")
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
stamp("3/7 matching + db + verificación")
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

# 4. Mapper incremental
stamp("4/7 mapper incremental (~2700 imágenes: horas)")
inc_out = test / "incremental"
inc_out.mkdir()
recs = pycolmap.incremental_mapping(db, timg, inc_out)
models = [(inc_out / str(k), v) for k, v in recs.items()]
assert models, "sin modelo"
sfm_path, best = max(models, key=lambda t: t[1].num_reg_images())
for k, r in sorted(recs.items(), key=lambda t: -t[1].num_reg_images())[:4]:
    print(f"modelo {k}: {r.num_reg_images()} cams, {r.num_points3D()} pts", flush=True)
print(f"MEJOR: {best.num_reg_images()}/{len(names)} registradas, "
      f"err {best.compute_mean_reprojection_error():.2f}px", flush=True)

# 5. Retriangulación + BA
stamp("5/7 retriangulación + BA")
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

# 6. Subset de entrenamiento + undistort + nube completa
stamp("6/7 subset + undistort")
reg = sorted(rec.reg_image_ids())
step = max(1, len(reg) // MAX_TRAIN)
keep = set(reg[::step])
for iid in reg:
    if iid not in keep:
        rec.deregister_frame(rec.image(iid).frame_id)
sfm_train = test / "sfm_train"
if sfm_train.exists():
    shutil.rmtree(sfm_train)
sfm_train.mkdir()
rec.write(str(sfm_train))
print(f"set de entrenamiento: {len(keep)} cámaras (1 de cada {step})", flush=True)

data = ROOT / "data_884full"
for d in (data / "images", data / "sparse", data / "stereo", data / "distorted"):
    if d.exists():
        shutil.rmtree(d)
pycolmap.undistort_images(output_path=str(data), input_path=str(sfm_train),
                          image_path=str(timg), output_type="COLMAP")
sparse0 = data / "sparse" / "0"
sparse0.mkdir(parents=True, exist_ok=True)
for f in (data / "sparse").iterdir():
    if f.is_file():
        shutil.move(str(f), sparse0 / f.name)
# nube inicial: TODOS los puntos del modelo retriangulado completo
shutil.copy(sfm_tri / "points3D.bin", sparse0 / "points3D.bin")

# 7. Entrenamiento
stamp(f"7/7 entrenamiento 3DGS ({ITERATIONS} iters, -r 4)")
r = subprocess.run(
    [PY, "train.py", "-s", str(data), "-m", str(MODEL_DIR),
     "-r", "4", "--data_device", "cpu",
     "--iterations", str(ITERATIONS),
     "--save_iterations", str(ITERATIONS), "--test_iterations", "-1"],
    cwd=ROOT / "gaussian-splatting")
if r.returncode:
    sys.exit(f"train.py falló con código {r.returncode}")
stamp(f"PIPELINE_DONE modelo en {MODEL_DIR}")
