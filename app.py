import os, time, base64, threading
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

_model = None
_model_name = None
_model_lock = threading.Lock()
_device = 'cpu'

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

# 全域錯誤 handler — 永遠回傳 JSON，不回傳 HTML
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
    global _model, _model_name, _device
    if 'model' not in request.files:
        return jsonify({'error': '未收到模型檔案'}), 400
    f = request.files['model']
    import tempfile, pathlib
    model_path = str(pathlib.Path(tempfile.gettempdir()) / f.filename)
    f.save(model_path)
    try:
        from ultralytics import YOLO
    except ImportError:
        return jsonify({'error': 'ultralytics 未安裝，請執行：pip install ultralytics'}), 500
    try:
        _device = get_device()
        with _model_lock:
            _model = YOLO(model_path)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _model.predict(dummy, device=_device, verbose=False)
            _model_name = f.filename
        return jsonify({'ok': True, 'model': f.filename, 'device': _device})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

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
        import cv2
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
                    'latency': latency_ms, 'device': _device, 'model': _model_name})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, ssl_context='adhoc')