"""Retriangulación completa del modelo GLOMAP + subset + undistort + 3DGS.

Reemplaza la retriangulación interna de GLOMAP (que crashea por el esquema
de db migrado) por pycolmap.triangulate_points sobre la db ORIGINAL de
esquema nuevo, que es 100% compatible. Produce la nube de puntos densa
(~300k) que GLOMAP habría dejado, con las mismas poses optimizadas.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pycolmap

ROOT = Path.home() / "dko-3dgs"
PY = str(ROOT / "gaussian-splatting" / "env" / "bin" / "python")
MAX_TRAIN_IMAGES = 500
ITERATIONS = 30000
MODEL_DIR = ROOT / "output" / "dko3d_full"

t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


inp = ROOT / "data" / "input"
db = ROOT / "hloc_out" / "sfm" / "database.db"  # la original, esquema nuevo
glomap_out = ROOT / "hloc_out" / "glomap"

models = [d for d in glomap_out.iterdir() if d.is_dir() and (d / "images.bin").exists()]
assert models, "no hay modelo GLOMAP"
sfm = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())

# 1. Retriangular todos los tracks con las poses ya optimizadas por GLOMAP.
#    Vía hloc.triangulation.main: crea una db NUEVA desde el propio modelo
#    (frames consistentes por construcción — triangulate_points directo choca
#    con el mismo check de frames que la retriangulación interna de GLOMAP).
stamp("1/5 retriangulación (hloc.triangulation sobre el modelo GLOMAP)")
from hloc import triangulation  # noqa: E402

rec = pycolmap.Reconstruction(str(sfm))
print(f"modelo GLOMAP ({sfm.name}): {rec.num_reg_images()} registradas, "
      f"{rec.num_points3D()} puntos antes de retriangular", flush=True)

out = ROOT / "hloc_out"

# pairs.txt tiene los 23916 nombres; la db creada desde el modelo solo conoce
# las imágenes registradas -> filtrar pares a registradas-con-registradas
reg_names = {rec.image(i).name for i in rec.reg_image_ids()}
pairs_tri = out / "pairs_tri.txt"
kept_pairs = []
for line in (out / "pairs.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) == 2 and parts[0] in reg_names and parts[1] in reg_names:
        kept_pairs.append(line)
pairs_tri.write_text("\n".join(kept_pairs) + "\n")
print(f"pares filtrados: {len(kept_pairs)} (solo imágenes registradas)", flush=True)

sfm_tri = ROOT / "hloc_out" / "glomap_tri"
if sfm_tri.exists():
    shutil.rmtree(sfm_tri)
rec = triangulation.main(
    sfm_tri,
    sfm,
    inp,
    pairs_tri,
    out / "feats-aliked-n16.h5",
    out / "feats-aliked-n16_matches-aliked-lightglue_pairs.h5",
)
print(f"retriangulado: {rec.num_points3D()} puntos 3D", flush=True)

# 2. Bundle adjustment global post-retriangulación (mismo pulido que hace
#    GLOMAP internamente después de retriangular)
stamp("2/5 bundle adjustment global post-retriangulación")
ba_opts = pycolmap.BundleAdjustmentOptions()
ba_opts.print_summary = True
pycolmap.bundle_adjustment(rec, ba_opts)
rec.write(str(sfm_tri))
print(f"BA hecho: {rec.num_points3D()} puntos, "
      f"error de reproyección medio {rec.compute_mean_reprojection_error():.3f}px", flush=True)

# 3. Submuestrear cámaras para entrenamiento
stamp("3/5 filtrar modelo a MAX_TRAIN_IMAGES cámaras")
reg = sorted(rec.reg_image_ids())
step = max(1, len(reg) // MAX_TRAIN_IMAGES)
keep = set(reg[::step])
for iid in reg:
    if iid not in keep:
        rec.deregister_frame(rec.image(iid).frame_id)

sfm_train = ROOT / "hloc_out" / "sfm_train"
if sfm_train.exists():
    shutil.rmtree(sfm_train)
sfm_train.mkdir()
rec.write(str(sfm_train))
print(f"set de entrenamiento: {len(keep)} cámaras (1 de cada {step})", flush=True)

# 3. Undistort al layout 3DGS + nube de puntos del modelo retriangulado completo
stamp("4/5 undistort + layout 3DGS")
data = ROOT / "data"
for d in (data / "images", data / "sparse", data / "stereo", data / "distorted"):
    if d.exists():
        shutil.rmtree(d)

pycolmap.undistort_images(
    output_path=str(data),
    input_path=str(sfm_train),
    image_path=str(inp),
    output_type="COLMAP",
)
sparse0 = data / "sparse" / "0"
sparse0.mkdir(parents=True, exist_ok=True)
for f in (data / "sparse").iterdir():
    if f.is_file():
        shutil.move(str(f), sparse0 / f.name)
# 3DGS solo lee xyz/rgb de points3D.bin: usar la nube retriangulada COMPLETA
shutil.copy(sfm_tri / "points3D.bin", sparse0 / "points3D.bin")

# 4. Entrenar
stamp(f"5/5 entrenamiento 3DGS ({ITERATIONS} iters, -r 2, data en RAM)")
r = subprocess.run(
    [PY, "train.py", "-s", str(data), "-m", str(MODEL_DIR),
     "-r", "2", "--data_device", "cpu",
     "--iterations", str(ITERATIONS),
     "--save_iterations", str(ITERATIONS),
     "--test_iterations", "-1"],
    cwd=ROOT / "gaussian-splatting",
)
if r.returncode:
    sys.exit(f"train.py falló con código {r.returncode}")

stamp(f"PIPELINE_DONE modelo en {MODEL_DIR}")
