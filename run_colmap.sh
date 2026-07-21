#!/bin/bash
# COLMAP SfM pipeline for dko walkthrough (CPU build, sequential video matching)
set -e
cd ~/dko-3dgs/data
mkdir -p distorted/sparse

echo "=== [1/5] feature extraction $(date +%T) ==="
colmap feature_extractor \
    --database_path distorted/database.db \
    --image_path input \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 0

echo "=== [2/5] sequential matching $(date +%T) ==="
colmap sequential_matcher \
    --database_path distorted/database.db \
    --SiftMatching.use_gpu 0 \
    --SequentialMatching.overlap 20 \
    --SequentialMatching.loop_detection 1 \
    --SequentialMatching.vocab_tree_path ~/dko-3dgs/vocab_tree.bin

echo "=== [3/5] mapper $(date +%T) ==="
colmap mapper \
    --database_path distorted/database.db \
    --image_path input \
    --output_path distorted/sparse \
    --Mapper.ba_global_function_tolerance=0.000001

echo "=== [4/5] undistort $(date +%T) ==="
colmap image_undistorter \
    --image_path input \
    --input_path distorted/sparse/0 \
    --output_path . \
    --output_type COLMAP

echo "=== [5/5] layout for 3DGS $(date +%T) ==="
mkdir -p sparse/0
find sparse -maxdepth 1 -type f -exec mv {} sparse/0/ \;

echo "=== COLMAP_DONE $(date +%T) ==="
