"""Reanudación tras corrupción de database.db por disco lleno.

Reconstruye la base de datos COLMAP desde los h5 ya computados (features +
matches, intactos), corre la verificación geométrica y encadena con
run_glomap_continue.py (GLOMAP -> subset -> undistort -> 3DGS).

Idempotente: si la db ya tiene las imágenes importadas, las reutiliza y
solo limpia keypoints/matches antes de reimportarlos.
"""
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pycolmap
from hloc.reconstruction import create_empty_db, get_image_ids, import_images
from hloc.triangulation import (estimation_and_geometric_verification,
                                import_features, import_matches)

ROOT = Path.home() / "dko-3dgs"
out = ROOT / "hloc_out"
images = ROOT / "data" / "input"
features = out / "feats-aliked-n16.h5"
matches = out / "feats-aliked-n16_matches-aliked-lightglue_pairs.h5"
pairs = out / "pairs.txt"
sfm = out / "sfm"
db = sfm / "database.db"

N_IMAGES = len(list(images.glob("*.jpg")))
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


stamp("1/3 preparar database.db")
reuse = False
if db.exists():
    try:
        con = sqlite3.connect(db)
        n = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        if n == N_IMAGES:
            for table in ("keypoints", "descriptors", "matches", "two_view_geometries"):
                con.execute(f"DELETE FROM {table}")
            con.commit()
            reuse = True
            print(f"db reutilizada ({n} imágenes ya importadas); tablas derivadas limpiadas", flush=True)
        con.close()
    except sqlite3.DatabaseError as e:
        print(f"db inutilizable ({e}); se recrea", flush=True)

if not reuse:
    for f in (db, db.with_name(db.name + "-shm"), db.with_name(db.name + "-wal")):
        if f.exists():
            f.unlink()
    create_empty_db(db)
    import_images(images, db, pycolmap.CameraMode.SINGLE, options=dict(camera_model="OPENCV"))

image_ids = get_image_ids(db)
print(f"{len(image_ids)} imágenes en la db", flush=True)
with pycolmap.Database.open(db) as dbh:
    import_features(image_ids, dbh, features)
    import_matches(image_ids, dbh, pairs, matches,
                   min_match_score=None, skip_geometric_verification=False)

stamp("2/3 verificación geométrica")
estimation_and_geometric_verification(db, pairs)

stamp("3/3 continuar con GLOMAP -> subset -> undistort -> 3DGS")
r = subprocess.run([str(ROOT / "gaussian-splatting" / "env" / "bin" / "python"),
                    str(ROOT / "run_glomap_continue.py")])
sys.exit(r.returncode)
