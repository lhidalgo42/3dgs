"""Test end-to-end del video nuevo IMG_0884.MOV (4K30 SDR, sin estabilización).

Extrae frames -> selección nítida ~500 -> seq+NetVLAD -> matching -> GLOMAP e
incremental -> diagnóstico (test884_diag.png). GATE automático: si el mejor
modelo registra >=80% con intrínsecos sanos, continúa con retriangulación,
BA, undistort y entrenamiento -> output/dko3d_test884. Si no, se detiene.
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
VIDEO = ROOT / "IMG_0884.MOV"

N_TEST = 500
OVERLAP = 15
RETRIEVAL_K = 20
ITERATIONS = 30000

cand = ROOT / "candidates884"
test = ROOT / "hloc_884"
timg = ROOT / "data_884" / "input"
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


# 1. Extraer todos los frames (si no están ya)
stamp("1/7 extracción de frames")
if not cand.exists() or len(list(cand.glob("*.jpg"))) < 26000:
    if cand.exists():
        shutil.rmtree(cand)
    cand.mkdir()
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", str(VIDEO), "-qscale:v", "2", str(cand / "%05d.jpg")])
    if r.returncode:
        sys.exit("ffmpeg falló")
frames = sorted(cand.glob("*.jpg"))
print(f"{len(frames)} frames", flush=True)

# 2. Selección nítida
stamp("2/7 selección de nítidos")
win = max(1, len(frames) // N_TEST)
if timg.exists():
    shutil.rmtree(timg)
timg.mkdir(parents=True)
for i in range(0, len(frames), win):
    window = frames[i:i + win]
    best = max(window, key=lambda f: cv2.Laplacian(
        cv2.imread(str(f), cv2.IMREAD_REDUCED_GRAYSCALE_4), cv2.CV_64F).var())
    os.link(best, timg / best.name)
names = sorted(p.name for p in timg.glob("*.jpg"))
print(f"{len(names)} seleccionados (ventana {win})", flush=True)

if test.exists():
    shutil.rmtree(test)
test.mkdir()

# 3. Features + pares
stamp("3/7 features ALIKED + NetVLAD + pares")
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

# 4. Matching + db + verificación
stamp("4/7 matching + db + verificación")
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

# 5. GLOMAP + incremental
stamp("5/7 GLOMAP + incremental")
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
r = subprocess.run([GLOMAP, "mapper", "--database_path", str(db_g),
                    "--image_path", str(timg), "--output_path", str(test / "glomap"),
                    "--skip_retriangulation", "1"])
glomap_models = []
if r.returncode == 0:
    for d in (test / "glomap").iterdir():
        if d.is_dir() and (d / "images.bin").exists():
            glomap_models.append((d, pycolmap.Reconstruction(str(d))))

inc_out = test / "incremental"
inc_out.mkdir()
recs = pycolmap.incremental_mapping(db, timg, inc_out)
inc_models = [(inc_out / str(k), v) for k, v in recs.items()]

# 6. Diagnóstico + gate
stamp("6/7 diagnóstico")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, models, title in [(axes[0], glomap_models, "GLOMAP"),
                          (axes[1], inc_models, "Incremental")]:
    if models:
        _, best = max(models, key=lambda t: t[1].num_reg_images())
        ids = sorted(best.reg_image_ids())
        c = np.array([best.image(i).projection_center() for i in ids])
        ax.scatter(c[:, 0], c[:, 2], s=5, c=np.arange(len(ids)), cmap="rainbow")
        cam = list(best.cameras.values())[0]
        ax.set_title(f"{title}: {best.num_reg_images()}/{len(names)} cams, "
                     f"{len(models)} modelos\nf={cam.params[0]:.0f} "
                     f"k1={cam.params[4]:.3f} p1={cam.params[6]:.4f}")
        ax.set_aspect("equal")
    else:
        ax.set_title(f"{title}: sin modelo")
plt.tight_layout()
plt.savefig(ROOT / "test884_diag.png", dpi=90)

print("===== RESUMEN =====", flush=True)
for tag, models in [("GLOMAP", glomap_models), ("INCREMENTAL", inc_models)]:
    for _, m in sorted(models, key=lambda t: -t[1].num_reg_images()):
        print(f"{tag}: {m.num_reg_images()} cams, {m.num_points3D()} pts, "
              f"err {m.compute_mean_reprojection_error():.2f}px", flush=True)


def sane(rec):
    cam = list(rec.cameras.values())[0]
    f = cam.params[0]
    return (rec.num_reg_images() >= 0.8 * len(names)
            and 1200 < f < 4500
            and abs(cam.params[4]) < 1.0 and abs(cam.params[6]) < 0.02)


candidates = [(d, m, tag) for tag, mods in [("glomap", glomap_models),
                                            ("incremental", inc_models)]
              for d, m in mods if sane(m)]
if not candidates:
    stamp("GATE_FAIL: ningún modelo pasa el gate (>=80% y intrínsecos sanos) — me detengo")
    sys.exit(0)

sfm_path, best, tag = max(candidates, key=lambda t: t[1].num_reg_images())
stamp(f"GATE_OK: {tag} con {best.num_reg_images()} cámaras — continúo a entrenamiento")

# 7. Retriangulación + BA + undistort + entrenamiento
stamp("7/7 retriangulación + BA + entrenamiento")
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

data = ROOT / "data_884"
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

MODEL_DIR = ROOT / "output" / "dko3d_test884"
r = subprocess.run(
    [PY, "train.py", "-s", str(data), "-m", str(MODEL_DIR),
     "-r", "2", "--data_device", "cpu",
     "--iterations", str(ITERATIONS),
     "--save_iterations", str(ITERATIONS), "--test_iterations", "-1"],
    cwd=ROOT / "gaussian-splatting")
if r.returncode:
    sys.exit(f"train.py falló con código {r.returncode}")
stamp(f"PIPELINE_DONE modelo en {MODEL_DIR}")
