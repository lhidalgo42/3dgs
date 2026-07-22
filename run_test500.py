"""Test chico end-to-end: 500 frames nítidos de todo el video con la receta nueva.

Receta: selección nítida 1-de-48 -> pares secuenciales + retrieval NetVLAD
(loop closures) -> matching (reutiliza features ALIKED ya extraídos) ->
GLOMAP -> diagnóstico de trayectoria -> retriangulación -> BA -> 3DGS.

Salidas: hloc_test/ (SfM), data_test/ (layout 3DGS), output/dko3d_test500,
         test_diag.png (trayectoria, para revisar ANTES de confiar en el modelo).
"""
import os
import shutil
import sqlite3
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
GLOMAP = str(Path.home() / "glomap-env" / "bin" / "glomap")

N_TEST = 500
OVERLAP = 15
RETRIEVAL_K = 20
ITERATIONS = 30000

src_images = ROOT / "data" / "input"          # 23916 hardlinks (nombres = claves h5)
feats_h5 = ROOT / "hloc_out" / "feats-aliked-n16.h5"
test = ROOT / "hloc_test"
timg = ROOT / "data_test" / "input"
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


# 1. Selección de 500 nítidos (Laplaciano a 1/4 de resolución, ventanas uniformes)
stamp("1/8 selección de frames nítidos")
frames = sorted(src_images.glob("*.jpg"))
assert frames, "no hay frames en data/input"
win = max(1, len(frames) // N_TEST)
if timg.exists():
    shutil.rmtree(timg)
timg.mkdir(parents=True)
kept = []
for i in range(0, len(frames), win):
    window = frames[i:i + win]
    best = max(window, key=lambda f: cv2.Laplacian(
        cv2.imread(str(f), cv2.IMREAD_REDUCED_GRAYSCALE_4), cv2.CV_64F).var())
    os.link(best, timg / best.name)  # mismo nombre -> mismas claves en el h5
    kept.append(best.name)
print(f"{len(kept)} frames seleccionados (ventana {win})", flush=True)

if test.exists():
    shutil.rmtree(test)
test.mkdir()

# 2. Pares: secuenciales + retrieval NetVLAD (loop closures)
stamp("2/8 pares secuenciales + NetVLAD")
names = sorted(kept)
pairs_all = set()
for i, a in enumerate(names):
    for j in range(i + 1, min(i + 1 + OVERLAP, len(names))):
        pairs_all.add((a, names[j]))

netvlad_conf = extract_features.confs["netvlad"]
global_desc = extract_features.main(netvlad_conf, timg, test)
pairs_retr = test / "pairs_retrieval.txt"
pairs_from_retrieval.main(global_desc, pairs_retr, num_matched=RETRIEVAL_K)
n_seq = len(pairs_all)
for line in pairs_retr.read_text().splitlines():
    a, b = line.split()
    if a != b and (b, a) not in pairs_all:
        pairs_all.add((a, b))
pairs_path = test / "pairs.txt"
pairs_path.write_text("\n".join(f"{a} {b}" for a, b in sorted(pairs_all)) + "\n")
print(f"pares: {n_seq} secuenciales + retrieval = {len(pairs_all)} totales", flush=True)

# 3. Matching (reutiliza los features ALIKED de las 23916)
stamp("3/8 matching ALIKED+LightGlue")
match_conf = match_features.confs["aliked+lightglue"]
matches_h5 = test / "matches.h5"
match_features.main(match_conf, pairs_path, feats_h5, matches=matches_h5)

# 4. Base de datos + verificación geométrica
stamp("4/8 base de datos + verificación geométrica")
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

# 5. Migrar db al esquema del glomap de conda-forge y correr GLOMAP
stamp("5/8 GLOMAP")
tpl = test / "db_template.db"
tpl.touch()
subprocess.run([GLOMAP, "mapper", "--database_path", str(tpl),
                "--image_path", "/nonexistent", "--output_path", str(test / "x")],
               capture_output=True)
db_g = sfm_dir / "database_glomap.db"
shutil.copy(tpl, db_g)
con = sqlite3.connect(db_g)
con.execute("ATTACH DATABASE ? AS src", (str(db),))
for t, cols in [
    ("cameras", "camera_id, model, width, height, params, prior_focal_length"),
    ("images", "image_id, name, camera_id"),
    ("keypoints", "image_id, rows, cols, data"),
    ("matches", "pair_id, rows, cols, data"),
    ("two_view_geometries", "pair_id, rows, cols, data, config, F, E, H, qvec, tvec"),
]:
    con.execute(f"INSERT INTO main.{t} ({cols}) SELECT {cols} FROM src.{t}")
con.commit()
con.close()

glomap_out = test / "glomap"
r = subprocess.run([GLOMAP, "mapper", "--database_path", str(db_g),
                    "--image_path", str(timg), "--output_path", str(glomap_out),
                    "--skip_retriangulation", "1"])
if r.returncode:
    sys.exit(f"glomap falló con código {r.returncode}")

models = [d for d in glomap_out.iterdir() if d.is_dir() and (d / "images.bin").exists()]
sfm = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
rec = pycolmap.Reconstruction(str(sfm))
print(f"GLOMAP: {rec.num_reg_images()}/{len(kept)} registradas, "
      f"{rec.num_points3D()} puntos", flush=True)

# 6. Diagnóstico de trayectoria (revisar test_diag.png antes de confiar)
stamp("6/8 diagnóstico de trayectoria")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ids = sorted(rec.reg_image_ids())
centers = np.array([rec.image(i).projection_center() for i in ids])
cam = list(rec.cameras.values())[0]
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, (a, b), name in zip(axes, [(0, 2), (0, 1), (1, 2)], ["X-Z", "X-Y", "Y-Z"]):
    sc = ax.scatter(centers[:, a], centers[:, b], s=4, c=np.arange(len(centers)), cmap="rainbow")
    ax.set_title(name)
    ax.set_aspect("equal")
plt.suptitle(f"test500: {len(ids)} cámaras | intrínsecos: {np.round(cam.params, 3)}")
plt.colorbar(sc, ax=axes[-1], label="orden temporal")
plt.tight_layout()
plt.savefig(ROOT / "test_diag.png", dpi=90)
print("test_diag.png guardado; intrínsecos:", cam.params, flush=True)

# 7. Retriangulación + BA global
stamp("7/8 retriangulación + BA")
reg_names = {rec.image(i).name for i in rec.reg_image_ids()}
pairs_tri = test / "pairs_tri.txt"
pairs_tri.write_text("\n".join(
    f"{a} {b}" for a, b in sorted(pairs_all) if a in reg_names and b in reg_names) + "\n")
sfm_tri = test / "tri"
rec = triangulation.main(sfm_tri, sfm, timg, pairs_tri, feats_h5, matches_h5)
print(f"retriangulado: {rec.num_points3D()} puntos", flush=True)
ba_opts = pycolmap.BundleAdjustmentOptions()
pycolmap.bundle_adjustment(rec, ba_opts)
rec.write(str(sfm_tri))
print(f"BA: error reproyección medio {rec.compute_mean_reprojection_error():.3f}px", flush=True)

# 8. Undistort + entrenamiento (todas las registradas, sin submuestreo)
stamp("8/8 undistort + entrenamiento 3DGS")
data_test = ROOT / "data_test"
for d in (data_test / "images", data_test / "sparse", data_test / "stereo", data_test / "distorted"):
    if d.exists():
        shutil.rmtree(d)
pycolmap.undistort_images(
    output_path=str(data_test), input_path=str(sfm_tri),
    image_path=str(timg), output_type="COLMAP",
)
sparse0 = data_test / "sparse" / "0"
sparse0.mkdir(parents=True, exist_ok=True)
for f in (data_test / "sparse").iterdir():
    if f.is_file():
        shutil.move(str(f), sparse0 / f.name)

MODEL_DIR = ROOT / "output" / "dko3d_test500"
r = subprocess.run(
    [PY, "train.py", "-s", str(data_test), "-m", str(MODEL_DIR),
     "-r", "2", "--data_device", "cpu",
     "--iterations", str(ITERATIONS),
     "--save_iterations", str(ITERATIONS),
     "--test_iterations", "-1"],
    cwd=ROOT / "gaussian-splatting",
)
if r.returncode:
    sys.exit(f"train.py falló con código {r.returncode}")

stamp(f"PIPELINE_DONE modelo de test en {MODEL_DIR}")
