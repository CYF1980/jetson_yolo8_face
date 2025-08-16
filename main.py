# main.py
import cv2
import time
import argparse
import numpy as np

from face_detector_yolov8 import YoloFaceDetector

def draw_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

def set_camera(cap, width, height, fps):
    # 盡力設定常見參數；有些攝影機/驅動會忽略
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,         fps)
    # 嘗試使用 MJPG，可提升 USB 視訊頻寬效率
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

def maybe_enable_cuda(net):
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            # DLA 通常不可用就跳過；FP16 對 Jetson 蠻有感
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("[INFO] OpenCV DNN 使用 CUDA FP16")
            return True
    except Exception as e:
        print(f"[WARN] 啟用 CUDA 失敗：{e}")
    print("[INFO] 改用 CPU 推論")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="USB 攝影機索引（/dev/video*）")
    ap.add_argument("--width",  type=int, default=640, help="攝影機寬度")
    ap.add_argument("--height", type=int, default=480, help="攝影機高度")
    ap.add_argument("--fps",    type=int, default=30,  help="攝影機影格率")
    ap.add_argument("--modelpath", type=str, default="weights/yolov8n-face.onnx", help="ONNX 權重路徑")
    ap.add_argument("--confThreshold", type=float, default=0.45, help="偵測信心門檻")
    ap.add_argument("--nmsThreshold",  type=float, default=0.5,  help="NMS IoU 門檻")
    ap.add_argument("--use_cuda", action="store_true", help="若可用則啟用 CUDA FP16")
    args = ap.parse_args()

    # 開啟 USB 攝影機
    cap = cv2.VideoCapture(args.device)
    set_camera(cap, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("[ERROR] 無法開啟攝影機"); return

    # 建立偵測器（與 main.py 相同類別）
    detector = YoloFaceDetector(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    if args.use_cuda:
        maybe_enable_cuda(detector.net)

    win_name = "YOLO Face 5pt Landmarks (USB cam)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # FPS（指數移動平均）
    prev_t = time.time()
    fps_ema = 0.0
    alpha = 0.9  # 越大越平滑

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 讀取到空畫面，結束"); break

            # 推論
            boxes, scores, classids, kpts = detector.detect(frame)

            # 繪圖
            vis = frame.copy()
            if boxes.size > 0:
                vis = detector.draw_detections(vis, boxes, scores, kpts)

            # FPS
            now = time.time()
            inst = 1.0 / max(1e-6, now - prev_t)
            prev_t = now
            fps_ema = inst if fps_ema == 0 else alpha * fps_ema + (1 - alpha) * inst
            draw_fps(vis, fps_ema)

            cv2.imshow(win_name, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
