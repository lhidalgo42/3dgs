"""Test A: mismos 509 frames del test anterior pero tone-mapeados HDR->SDR.

Extrae los frames con zscale+tonemap (hable), features ALIKED frescos,
pares secuenciales+NetVLAD, matching, y corre GLOMAP e incremental para
comparar contra los resultados con frames HDR lavados. Se detiene tras el
diagnóstico (testA_diag.png) — sin entrenamiento.
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
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.reconstruction import create_empty_db, get_image_ids, import_images
from hloc.triangulation import (estimation_and_geometric_verification,
                                import_features, import_matches)

ROOT = Path.home() / "dko-3dgs"
GLOMAP = str(Path.home() / "glomap-env" / "bin" / "glomap")
VIDEO = ROOT / "dko.mov"
OVERLAP = 15
RETRIEVAL_K = 20

test = ROOT / "hloc_testA"
timg = ROOT / "data_testA" / "input"
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


# 1. Extraer los mismos 509 frames, tone-mapeados (nombre = frame n del video)
stamp("1/6 extracción tone-mapped de los 509 frames")
names = sorted(p.name for p in (ROOT / "data_test" / "input").glob("*.jpg"))
assert names, "no existe la selección del test anterior"
frame_ids = [int(Path(n).stem) for n in names]

if timg.exists():
    shutil.rmtree(timg)
timg.mkdir(parents=True)
sel = "+".join(f"eq(n\\,{i})" for i in frame_ids)
vf = (f"select='{sel}',zscale=transfer=linear:npl=100,"
      "tonemap=tonemap=hable:desat=0,"
      "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv,format=yuv420p")
r = subprocess.run(
    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(VIDEO),
     "-vf", vf, "-fps_mode", "vfr", "-qscale:v", "2",
     str(timg / "tmp_%05d.jpg")])
if r.returncode:
    sys.exit("ffmpeg falló")
tmp = sorted(timg.glob("tmp_*.jpg"))
assert len(tmp) == len(names), f"extraídos {len(tmp)} != {len(names)}"
for t, n in zip(tmp, names):
    t.rename(timg / n)
print(f"{len(names)} frames tone-mapeados", flush=True)

if test.exists():
    shutil.rmtree(test)
test.mkdir()

# 2. Features frescos (ALIKED + NetVLAD) sobre los frames tone-mapeados
stamp("2/6 features ALIKED + NetVLAD")
feat_conf = extract_features.confs["aliked-n16"]
feats_h5 = extract_features.main(feat_conf, timg, test)
global_desc = extract_features.main(extract_features.confs["netvlad"], timg, test)

# 3. Pares secuenciales + retrieval
stamp("3/6 pares")
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
stamp("4/6 matching + db + verificación")
match_conf = match_features.confs["aliked+lightglue"]
matches_h5 = test / "matches.h5"
match_features.main(match_conf, pairs_path, feats_h5, matches=matches_h5)

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

# 5. GLOMAP + incremental en paralelo conceptual (secuencial aquí)
stamp("5/6 GLOMAP")
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
            glomap_models.append(pycolmap.Reconstruction(str(d)))
else:
    print(f"glomap falló ({r.returncode}) — sigo con incremental", flush=True)

stamp("5/6b mapper incremental")
inc_out = test / "incremental"
inc_out.mkdir()
recs = pycolmap.incremental_mapping(db, timg, inc_out)
inc_models = list(recs.values())

# 6. Diagnóstico comparativo
stamp("6/6 diagnóstico")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, models, title in [
    (axes[0], glomap_models, "GLOMAP"),
    (axes[1], inc_models, "Incremental"),
]:
    best = max(models, key=lambda r: r.num_reg_images()) if models else None
    if best:
        ids = sorted(best.reg_image_ids())
        c = np.array([best.image(i).projection_center() for i in ids])
        sc = ax.scatter(c[:, 0], c[:, 2], s=5, c=np.arange(len(ids)), cmap="rainbow")
        cam = list(best.cameras.values())[0]
        ax.set_title(f"{title}: mejor modelo {best.num_reg_images()}/{len(names)} cams, "
                     f"{len(models)} modelos\nf={cam.params[0]:.0f},{cam.params[1]:.0f} "
                     f"k1={cam.params[4]:.3f} p1={cam.params[6]:.4f}")
        ax.set_aspect("equal")
    else:
        ax.set_title(f"{title}: sin modelo")
plt.tight_layout()
plt.savefig(ROOT / "testA_diag.png", dpi=90)

print("\n===== RESUMEN =====", flush=True)
for tag, models in [("GLOMAP", glomap_models), ("INCREMENTAL", inc_models)]:
    for m in sorted(models, key=lambda r: -r.num_reg_images()):
        print(f"{tag}: {m.num_reg_images()} cams, {m.num_points3D()} pts, "
              f"err {m.compute_mean_reprojection_error():.2f}px", flush=True)
stamp("TESTA_DONE diagnóstico en testA_diag.png")
