"""Continuación del pipeline: GLOMAP (mapper global) -> subset -> undistort -> 3DGS.

Se lanza cuando hloc ya dejó lista la base de datos (features + matches +
verificación geométrica) en hloc_out/sfm/database.db, en lugar del mapper
incremental de pycolmap que tomaría días con ~24k imágenes.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pycolmap

ROOT = Path.home() / "dko-3dgs"
PY = str(ROOT / "gaussian-splatting" / "env" / "bin" / "python")
GLOMAP = str(Path.home() / "glomap-env" / "bin" / "glomap")
MAX_TRAIN_IMAGES = 500
ITERATIONS = 30000
MODEL_DIR = ROOT / "output" / "dko3d_full"

t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


inp = ROOT / "data" / "input"
db = ROOT / "hloc_out" / "sfm" / "database.db"
assert db.exists(), f"no existe {db}"

# 1. GLOMAP mapper global
stamp("1/4 GLOMAP mapper global")
glomap_out = ROOT / "hloc_out" / "glomap"
if glomap_out.exists():
    shutil.rmtree(glomap_out)
r = subprocess.run([GLOMAP, "mapper",
                    "--database_path", str(db),
                    "--image_path", str(inp),
                    "--output_path", str(glomap_out)])
if r.returncode:
    sys.exit(f"glomap falló con código {r.returncode}")

models = [d for d in glomap_out.iterdir() if d.is_dir() and (d / "images.bin").exists()]
assert models, "glomap no produjo ningún modelo"
sfm = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())

# 2. Submuestrear cámaras para entrenamiento
stamp("2/4 filtrar modelo a MAX_TRAIN_IMAGES cámaras")
rec = pycolmap.Reconstruction(str(sfm))
reg = sorted(rec.reg_image_ids())
print(f"modelo GLOMAP ({sfm.name}): {len(reg)} registradas, {rec.num_points3D()} puntos", flush=True)

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

# 3. Undistort al layout 3DGS + nube de puntos del modelo completo
stamp("3/4 undistort + layout 3DGS")
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
# 3DGS solo lee xyz/rgb de points3D.bin: usar la nube del modelo COMPLETO
shutil.copy(sfm / "points3D.bin", sparse0 / "points3D.bin")

# 4. Entrenar
stamp(f"4/4 entrenamiento 3DGS ({ITERATIONS} iters, -r 2, data en RAM)")
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
