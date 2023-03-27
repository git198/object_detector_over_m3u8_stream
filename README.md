# object_detector_over_m3u8_stream

This is a short project that:
    - Captures a webcam stream 
    - runs 2 DNN models in parallel using ray 
    - shows detections

Setup:

sudo apt install ffmpeg
yes | conda create -n hyperfly python=3.8
conda activate hyperfly
yes | pip install -r requirements.txt

python _2_ray_run_and_show.py