# dko-3dgs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhidalgo42/3dgs/blob/main/dko_3dgs_colab.ipynb)

Reconstrucción 3D de un recorrido (walkthrough en video) usando **3D Gaussian Splatting**.

Dos notebooks para correr el pipeline completo:

- [`dko_3dgs_colab.ipynb`](dko_3dgs_colab.ipynb) — en Google Colab (GPU T4, video desde Google Drive), abre con el badge de arriba.
- [`dko_3dgs_local.ipynb`](dko_3dgs_local.ipynb) — en local (`~/dko-3dgs`), reutiliza los scripts del repo y los datos ya existentes.

Pipeline completo: video → selección de frames nítidos → SfM (COLMAP o hloc) → entrenamiento 3DGS.

## Pipeline

```
video → frames (candidates/) → select_sharp.py → data/input/
      → run_colmap.sh  (o run_hloc.py) → data/sparse/0 + data/images/
      → gaussian-splatting/train.py → output/dko3d/point_cloud/
```

### 1. Extracción y selección de frames

Extraer frames del video con ffmpeg a `candidates/`, luego quedarse con el más nítido (varianza del Laplaciano) de cada ventana de N frames:

```bash
python select_sharp.py candidates/ data/input/ 8
```

### 2. Structure from Motion

Dos alternativas que producen el mismo layout (`data/sparse/0` + `data/images`):

**COLMAP (CPU, SIFT + matching secuencial con loop detection):**

```bash
./run_colmap.sh
```

Requiere `vocab_tree.bin` en la raíz del proyecto (descargar de [demuc.de/colmap](https://demuc.de/colmap/)).

**hloc (GPU, ALIKED + LightGlue):**

```bash
python run_hloc.py
```

Genera la reconstrucción en `hloc_out/sfm/`.

### 3. Entrenamiento 3DGS

Con el repo de [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) clonado en `gaussian-splatting/`:

```bash
cd gaussian-splatting
python train.py -s ../data -m ../output/dko3d
```

El modelo entrenado queda en `output/dko3d/point_cloud/iteration_*/point_cloud.ply`, visualizable con SIBR viewer o cualquier visor de splats (p. ej. [SuperSplat](https://playcanvas.com/supersplat/editor)).

## Estructura del proyecto

| Ruta | Contenido | En git |
|---|---|---|
| `select_sharp.py` | Selección de frames nítidos por ventana | ✅ |
| `run_colmap.sh` | Pipeline SfM con COLMAP (CPU) | ✅ |
| `run_hloc.py` | Pipeline SfM con hloc/pycolmap (GPU) | ✅ |
| `candidates/` | Frames extraídos del video (~2000) | ❌ |
| `data/` | Dataset en formato 3DGS (input, images, sparse) | ❌ |
| `hloc_out/` | Features, matches y SfM de hloc | ❌ |
| `output/` | Modelos 3DGS entrenados | ❌ |
| `gaussian-splatting/` | Clon del repo de Inria | ❌ |

Los datos pesados (imágenes, reconstrucciones, modelos) están excluidos por `.gitignore`; el repo versiona solo scripts y documentación.

## Requisitos

- COLMAP (build CPU sirve) **o** [hloc](https://github.com/cvg/Hierarchical-Localization) + pycolmap con GPU
- Python: `opencv-python`, `pycolmap`, `hloc`
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) con sus submódulos CUDA para entrenar
- ffmpeg para extraer frames
