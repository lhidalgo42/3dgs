"""IMG_0884 con TODOS los frames (26.754) — CON CHECKPOINTS reanudables.

Cada etapa deja un marcador en hloc_884all/ckpt/; al relanzar el script se
salta lo ya hecho. Features y matching además reanudan a mitad de etapa
(hloc salta lo ya presente en los .h5). El entrenamiento guarda checkpoints
de PyTorch cada 5000 iteraciones y se reanuda del último.

Pausar:   pkill -f 'python run_all884.py'  (o apagar el PC)
Reanudar: nohup gaussian-splatting/env/bin/python run_all884.py >> pipeline_884all.log 2>&1 &

Salida: output/dko3d_884all
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

OFFSETS = [1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200]
RETRIEVAL_K = 15
MAX_TRAIN = 2500
ITERATIONS = 30000
CKPT_ITERS = [5000, 10000, 15000, 20000, 25000]
MODEL_DIR = ROOT / "output" / "dko3d_884all"

cand = ROOT / "candidates884"
test = ROOT / "hloc_884all"
timg = ROOT / "data_884all" / "input"
ckpt = test / "ckpt"
test.mkdir(exist_ok=True)
ckpt.mkdir(exist_ok=True)
t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


def done(tag):
    return (ckpt / tag).exists()


def mark(tag):
    (ckpt / tag).touch()


# 1. Todos los frames
if not done("1_frames"):
    stamp("1/7 hardlinks de TODOS los frames")
    frames = sorted(cand.glob("*.jpg"))
    assert len(frames) > 26000, "faltan frames en candidates884"
    if timg.exists():
        shutil.rmtree(timg)
    timg.mkdir(parents=True)
    for f in frames:
        os.link(f, timg / f.name)
    mark("1_frames")
else:
    stamp("1/7 [ckpt] frames ya enlazados")
names = sorted(p.name for p in timg.glob("*.jpg"))
print(f"{len(names)} frames", flush=True)

# 2. Features (hloc reanuda solo: salta imágenes ya presentes en el h5)
stamp("2/7 features ALIKED + NetVLAD (reanudable a mitad)")
feats_h5 = extract_features.main(extract_features.confs["aliked-n16"], timg, test)
global_desc = extract_features.main(extract_features.confs["netvlad"], timg, test)
mark("2_features")

# 3. Pares
pairs_path = test / "pairs.txt"
if not done("3_pairs"):
    stamp("3/7 pares escalonados + retrieval")
    pairs_all = set()
    n = len(names)
    for i in range(n):
        for off in OFFSETS:
            if i + off < n:
                pairs_all.add((names[i], names[i + off]))
    n_seq = len(pairs_all)
    pairs_retr = test / "pairs_retrieval.txt"
    pairs_from_retrieval.main(global_desc, pairs_retr, num_matched=RETRIEVAL_K)
    for line in pairs_retr.read_text().splitlines():
        a, b = line.split()
        if a != b and (b, a) not in pairs_all:
            pairs_all.add((a, b))
    pairs_path.write_text("\n".join(f"{a} {b}" for a, b in sorted(pairs_all)) + "\n")
    print(f"pares: {n_seq} secuenciales escalonados -> {len(pairs_all)} totales", flush=True)
    mark("3_pairs")
else:
    stamp("3/7 [ckpt] pares ya generados")
pairs_all = set(tuple(l.split()) for l in pairs_path.read_text().splitlines() if l.strip())

# 4a. Matching (hloc reanuda solo: salta pares ya presentes en matches.h5)
matches_h5 = test / "matches.h5"
stamp(f"4/7 matching de {len(pairs_all)} pares (~8-10h GPU, reanudable a mitad)")
match_features.main(match_features.confs["aliked+lightglue"], pairs_path, feats_h5,
                    matches=matches_h5)
mark("4a_matching")

# 4b. db + verificación
sfm_dir = test / "sfm"
db = sfm_dir / "database.db"
if not done("4b_db"):
    stamp("4/7b base de datos + verificación geométrica")
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sfm_dir.mkdir()
    create_empty_db(db)
    import_images(timg, db, pycolmap.CameraMode.SINGLE, options=dict(camera_model="OPENCV"))
    image_ids = get_image_ids(db)
    with pycolmap.Database.open(db) as dbh:
        import_features(image_ids, dbh, feats_h5)
        import_matches(image_ids, dbh, pairs_path, matches_h5,
                       min_match_score=None, skip_geometric_verification=False)
    estimation_and_geometric_verification(db, pairs_path)
    mark("4b_db")
else:
    stamp("4/7b [ckpt] db ya verificada")

# 5. GLOMAP
glomap_out = test / "glomap"
if not done("5_glomap"):
    stamp("5/7 GLOMAP (sin checkpoint interno: si se corta, esta etapa se repite)")
    tpl = test / "db_template.db"
    if not tpl.exists():
        tpl.touch()
        subprocess.run([GLOMAP, "mapper", "--database_path", str(tpl),
                        "--image_path", "/nonexistent", "--output_path", str(test / "x")],
                       capture_output=True)
    db_g = sfm_dir / "database_glomap.db"
    if not db_g.exists():
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
    if glomap_out.exists():
        shutil.rmtree(glomap_out)
    r = subprocess.run([GLOMAP, "mapper", "--database_path", str(db_g),
                        "--image_path", str(timg), "--output_path", str(glomap_out),
                        "--skip_retriangulation", "1"])
    if r.returncode:
        sys.exit(f"glomap falló con código {r.returncode}")
    mark("5_glomap")
else:
    stamp("5/7 [ckpt] GLOMAP ya hecho")
models = [d for d in glomap_out.iterdir() if (d / "images.bin").exists()]
sfm_path = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
best = pycolmap.Reconstruction(str(sfm_path))
print(f"GLOMAP: {best.num_reg_images()}/{len(names)} registradas, "
      f"{best.num_points3D()} puntos", flush=True)

# 6. Retriangulación
sfm_tri = test / "tri"
if not done("6_tri"):
    stamp("6/7 retriangulación")
    reg_names = {best.image(i).name for i in best.reg_image_ids()}
    pairs_tri = test / "pairs_tri.txt"
    pairs_tri.write_text("\n".join(
        f"{a} {b}" for a, b in sorted(pairs_all)
        if a in reg_names and b in reg_names) + "\n")
    if sfm_tri.exists():
        shutil.rmtree(sfm_tri)
    rec = triangulation.main(sfm_tri, sfm_path, timg, pairs_tri, feats_h5, matches_h5)
    print(f"retriangulado: {rec.num_points3D()} puntos, "
          f"err {rec.compute_mean_reprojection_error():.3f}px", flush=True)
    mark("6_tri")
else:
    stamp("6/7 [ckpt] retriangulación ya hecha")
    rec = pycolmap.Reconstruction(str(sfm_tri))

# 7a. Subset + undistort
data = ROOT / "data_884all"
if not done("7a_undistort"):
    stamp("7/7a subset + undistort")
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
    shutil.copy(sfm_tri / "points3D.bin", sparse0 / "points3D.bin")
    mark("7a_undistort")
else:
    stamp("7/7a [ckpt] undistort ya hecho")

# 7b. Entrenamiento (reanuda del último checkpoint de PyTorch si existe)
stamp(f"7/7b entrenamiento 3DGS ({ITERATIONS} iters, -r 4, ckpt cada 5000)")
train_cmd = [PY, "train.py", "-s", str(data), "-m", str(MODEL_DIR),
             "-r", "4", "--data_device", "cpu",
             "--iterations", str(ITERATIONS),
             "--save_iterations", str(ITERATIONS), "--test_iterations", "-1",
             "--checkpoint_iterations"] + [str(i) for i in CKPT_ITERS]
cpts = sorted(MODEL_DIR.glob("chkpnt*.pth"),
              key=lambda p: int(p.stem.replace("chkpnt", ""))) if MODEL_DIR.exists() else []
if cpts:
    print(f"reanudando desde {cpts[-1].name}", flush=True)
    train_cmd += ["--start_checkpoint", str(cpts[-1])]
r = subprocess.run(train_cmd, cwd=ROOT / "gaussian-splatting")
if r.returncode:
    sys.exit(f"train.py falló con código {r.returncode}")
stamp(f"PIPELINE_DONE modelo en {MODEL_DIR}")
