#!/bin/bash

# Deactivate any existing conda environments
conda deactivate

# Delete the conda environment
conda remove --name mukh-pip --all -y

# Create a new conda environment
conda create -n mukh-pip python=3.10 -y

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mukh-pip
pip install mukh==0.1.13

echo "\nCleaning output directory..."
rm -rf output
echo "Output directory cleaned"
echo "--------------------------------"

echo "\nTesting deepfake detection pipeline..."
python -m examples.pipelines.deepfake_detection --media_path assets/images/img1.jpg --output_folder output/deepfake_detection_pipeline_img
python -m examples.pipelines.deepfake_detection --media_path data/demo_fake/elon_musk.mp4 --output_folder output/deepfake_detection_pipeline_video
echo "Completed testing deepfake detection pipeline"
echo "--------------------------------"

echo "\nTesting face detection..."
echo "--------------------------------"

echo "\nTesting blazeface..."
echo "--------------------------------"
python -m examples.face_detection.basic_detection --detection_model blazeface
echo "Completed testing blazeface"
echo "--------------------------------"

echo "\nTesting ultralight..."
echo "--------------------------------"
python -m examples.face_detection.basic_detection --detection_model ultralight
echo "Completed testing ultralight"
echo "--------------------------------"

echo "\nTesting mediapipe..."
python -m examples.face_detection.basic_detection --detection_model mediapipe
echo "Completed testing mediapipe"
echo "--------------------------------"

echo "\nTesting deepfake detection..."
python -m examples.deepfake_detection.detection --detection_model resnet_inception
python -m examples.deepfake_detection.detection --detection_model efficientnet
echo "Completed testing deepfake detection"
echo "--------------------------------"

echo "\nTesting reenactment..."
python -m examples.reenactment.basic_reenactment --reenactor_model tps
echo "Completed testing reenactment"
echo "--------------------------------"