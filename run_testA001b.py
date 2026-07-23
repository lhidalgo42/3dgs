"""Variante B del test Blackmagic: muestreo espaciado (1 de cada 4 del set A).

Hipótesis: el 50% de registro de la variante A se debe a frames demasiado
juntos en el tiempo (0.1s -> baselines chicos), no a la cámara. Aquí los
frames quedan a ~0.4s. Reutiliza features/NetVLAD de hloc_A001.
Solo SfM incremental + reporte — sin entrenamiento.
"""
import os
import shutil
import time
from pathlib import Path

import pycolmap
from hloc import match_features, pairs_from_retrieval
from hloc.reconstruction import create_empty_db, get_image_ids, import_images
from hloc.triangulation import (estimation_and_geometric_verification,
                                import_features, import_matches)

ROOT = Path.home() / "dko-3dgs"
srcA = ROOT / "hloc_A001"
feats_h5 = srcA / "feats-aliked-n16.h5"
global_desc = srcA / "global-feats-netvlad.h5"
OVERLAP = 15
RETRIEVAL_K = 20

test = ROOT / "hloc_A001b"
timg = ROOT / "data_A001b" / "input"
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


stamp("1/3 subset 1-de-4 + pares")
all_names = sorted(p.name for p in (ROOT / "data_A001" / "input").glob("*.jpg"))
names = all_names[::4]
if timg.exists():
    shutil.rmtree(timg)
timg.mkdir(parents=True)
for n in names:
    os.link(ROOT / "data_A001" / "input" / n, timg / n)
print(f"{len(names)} frames (de {len(all_names)})", flush=True)

if test.exists():
    shutil.rmtree(test)
test.mkdir()

pairs_all = set()
for i, a in enumerate(names):
    for j in range(i + 1, min(i + 1 + OVERLAP, len(names))):
        pairs_all.add((a, names[j]))
pairs_retr = test / "pairs_retrieval.txt"
pairs_from_retrieval.main(global_desc, pairs_retr, num_matched=RETRIEVAL_K,
                          query_list=names, db_list=names)
for line in pairs_retr.read_text().splitlines():
    a, b = line.split()
    if a != b and (b, a) not in pairs_all:
        pairs_all.add((a, b))
pairs_path = test / "pairs.txt"
pairs_path.write_text("\n".join(f"{a} {b}" for a, b in sorted(pairs_all)) + "\n")
print(f"{len(pairs_all)} pares", flush=True)

stamp("2/3 matching + db + verificación")
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

stamp("3/3 mapper incremental")
inc_out = test / "incremental"
inc_out.mkdir()
recs = pycolmap.incremental_mapping(db, timg, inc_out)
for k, r in sorted(recs.items(), key=lambda t: -t[1].num_reg_images()):
    print(f"modelo {k}: {r.num_reg_images()}/{len(names)} registradas, "
          f"{r.num_points3D()} pts, err {r.compute_mean_reprojection_error():.2f}px",
          flush=True)
stamp("B_DONE")
