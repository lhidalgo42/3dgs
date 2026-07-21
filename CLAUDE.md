# CLAUDE.md

Guía para Claude Code al trabajar en este repo.

## Qué es

Pipeline de reconstrucción 3D Gaussian Splatting a partir de un video walkthrough ("dko"). El repo versiona **solo scripts y docs**; los datos (frames, reconstrucciones SfM, modelos entrenados) viven localmente y están en `.gitignore`.

## Flujo de datos

1. `candidates/` — ~2000 frames extraídos del video con ffmpeg.
2. `select_sharp.py <src> <dst> <win>` — copia a `data/input/` el frame más nítido (varianza del Laplaciano) de cada ventana de `win` frames. Corrida actual: 240 imágenes seleccionadas.
3. SfM, dos rutas equivalentes:
   - `run_colmap.sh` — COLMAP CPU, SIFT, matching secuencial (overlap 20) con loop detection vía `vocab_tree.bin`. Escribe `data/distorted/`, undistorsiona a `data/images/` y deja el modelo en `data/sparse/0/` (layout que espera 3DGS).
   - `run_hloc.py` — hloc GPU, ALIKED n16 + LightGlue, pares secuenciales (overlap 15), mapper de pycolmap. Salida en `hloc_out/sfm/`.
4. Entrenamiento: `cd gaussian-splatting && python train.py -s ../data -m ../output/dko3d`. Última corrida: iteración 7000, 225 imágenes registradas.

## Convenciones y precauciones

- **Nunca** hacer `git add` de `candidates/`, `data/`, `hloc_out/`, `output/`, `gaussian-splatting/` ni `vocab_tree.bin` — son cientos de MB a GB.
- `gaussian-splatting/` es un clon de https://github.com/graphdeco-inria/gaussian-splatting con su propio `.git`; no modificarlo salvo pedido explícito.
- Los scripts asumen rutas bajo `~/dko-3dgs` (hardcodeadas en `run_colmap.sh` y `run_hloc.py`).
- COLMAP corre en CPU en esta máquina (WSL2); las etapas de extracción y matching son lentas — no matar procesos que parecen colgados, revisar `colmap.log`.
- Los logs de corridas largas van a `*.log` en la raíz (`colmap.log`, `hloc.log`), también ignorados por git.
- Formato de dataset 3DGS esperado: `data/images/` + `data/sparse/0/{cameras,images,points3D}.bin`.
