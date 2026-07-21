"""Pipeline completo con TODOS los frames: input -> hloc SfM -> subset -> undistort -> 3DGS.

Correr con el python del venv:
    nohup gaussian-splatting/env/bin/python run_full_local.py > pipeline_full.log 2>&1 &
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "dko-3dgs"
PY = str(ROOT / "gaussian-splatting" / "env" / "bin" / "python")
MAX_TRAIN_IMAGES = 500
ITERATIONS = 30000
MODEL_DIR = ROOT / "output" / "dko3d_full"

t0 = time.time()


def stamp(msg):
    print(f"=== [{time.strftime('%H:%M:%S')}] (+{(time.time() - t0) / 60:.0f}m) {msg}", flush=True)


# 1. data/input: hardlinks a todos los frames
stamp("1/5 hardlinks a data/input")
frames = sorted((ROOT / "candidates").glob("*.jpg"))
assert frames, "no hay frames en candidates/"
inp = ROOT / "data" / "input"
if inp.exists():
    shutil.rmtree(inp)
inp.mkdir(parents=True)
for i, f in enumerate(frames):
    os.link(f, inp / f"{i:05d}.jpg")
print(f"{len(frames)} frames enlazados", flush=True)

# 2. SfM con hloc (limpia la corrida anterior)
stamp("2/5 hloc SfM (extracción GPU + matching GPU + mapper CPU)")
if (ROOT / "hloc_out").exists():
    shutil.rmtree(ROOT / "hloc_out")
r = subprocess.run([PY, str(ROOT / "run_hloc.py")], cwd=ROOT)
if r.returncode:
    sys.exit(f"run_hloc.py falló con código {r.returncode}")

# 3. Submuestrear cámaras para entrenamiento
stamp("3/5 filtrar modelo a MAX_TRAIN_IMAGES cámaras")
import pycolmap  # noqa: E402  (import tardío: solo existe en el venv)

sfm = ROOT / "hloc_out" / "sfm"
rec = pycolmap.Reconstruction(str(sfm))
reg = sorted(rec.reg_image_ids())
print(f"modelo SfM: {len(reg)} registradas de {len(frames)}, {rec.num_points3D()} puntos", flush=True)

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

# 4. Undistort al layout 3DGS + nube de puntos del modelo completo
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
# 3DGS solo lee xyz/rgb de points3D.bin: usar la nube del modelo COMPLETO
# (al filtrar cámaras los tracks pierden observaciones y quedarían ~0 puntos)
shutil.copy(sfm / "points3D.bin", sparse0 / "points3D.bin")

# 5. Entrenar
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
