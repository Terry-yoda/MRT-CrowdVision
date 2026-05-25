import os, time, base64, threading, glob
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# ==========================================
# 系統設定 (請修改為你的 YOLOv12 權重路徑)
# ==========================================
DEFAULT_MODEL_PATH = "best.pt" 

_model = None
_device = 'cpu'
_model_lock = threading.Lock()

def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/device', methods=['GET'])
def api_device():
    d = get_device()
    info = {'device': d}
    try:
        import torch
        if d == 'cuda':
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    return jsonify(info)

@app.route('/api/load', methods=['POST'])
def api_load():
    global _model, _device
    try:
        from ultralytics import YOLO
    except ImportError:
        return jsonify({'error': 'ultralytics 未安裝，請執行：pip install ultralytics'}), 500
    try:
        _device = get_device()
        with _model_lock:
            _model = YOLO(DEFAULT_MODEL_PATH)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _model.predict(dummy, device=_device, verbose=False)
        return jsonify({'ok': True, 'model': DEFAULT_MODEL_PATH, 'device': _device})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/stream_folder', methods=['GET'])
def stream_folder():
    """將指定的資料夾內的圖片轉為 MJPEG 串流，並調整為 5 秒換一張"""
    folder = request.args.get('path', '')
    fps = float(request.args.get('fps', 0.2)) # 預設 0.2 FPS = 5 秒換一張
    
    def generate():
        if not os.path.isdir(folder):
            yield b''
            return
        
        while True:
            images = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
                images.extend(glob.glob(os.path.join(folder, ext)))
                images.extend(glob.glob(os.path.join(folder, ext.upper())))
            
            images = sorted(list(set(images)))
            if not images:
                yield b''
                time.sleep(2)
                continue
                
            for img_path in images:
                img = cv2.imread(img_path)
                if img is not None:
                    ret, jpeg = cv2.imencode('.jpg', img)
                    if ret:
                        frame = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(1.0 / fps) # fps=0.2 時，即為停止 5 秒
                
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/infer', methods=['POST'])
def api_infer():
    global _model
    if _model is None:
        return jsonify({'error': '模型尚未載入'}), 400
    data = request.get_json(force=True)
    img_b64  = data.get('image', '')
    conf_thr = float(data.get('conf', 0.5))
    iou_thr  = float(data.get('iou', 0.45))
    size     = int(data.get('size', 640))
    try:
        header, encoded = img_b64.split(',', 1) if ',' in img_b64 else ('', img_b64)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': '無法解碼影像'}), 400
    except Exception as e:
        return jsonify({'error': f'影像解碼失敗: {e}'}), 400
    try:
        t0 = time.perf_counter()
        with _model_lock:
            results = _model.predict(img, device=_device, conf=conf_thr,
                                     iou=iou_thr, imgsz=size, verbose=False)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        return jsonify({'error': f'推論失敗: {e}'}), 500
    
    h, w = img.shape[:2]
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = float(box.conf[0])
            boxes.append({'x': x1/w, 'y': y1/h, 'w': (x2-x1)/w, 'h': (y2-y1)/h, 'conf': c})
            
    return jsonify({'boxes': boxes, 'count': len(boxes),
                    'latency': latency_ms, 'device': _device, 'model': DEFAULT_MODEL_PATH})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, ssl_context='adhoc')