


python3 main.py --device 0 --width 640 --height 480 --fps 30 \
    --modelpath model/yolov8-lite-s.onnx \
    --confThreshold 0.45 \
    --nmsThreshold 0.5 \
    --use_cuda

python3 main.py --device 0 --width 640 --height 480 --fps 30 \
    --modelpath model/yolov8-lite-t.onnx \
    --confThreshold 0.45 \
    --nmsThreshold 0.5 \
    --use_cuda

python3 main.py --device 0 --width 640 --height 480 --fps 30 \
    --modelpath model/yolov8n-face.onnx \
    --confThreshold 0.45 \
    --nmsThreshold 0.5 \
    --use_cuda

