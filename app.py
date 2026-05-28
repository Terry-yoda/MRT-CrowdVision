import os, time, threading, glob, hashlib, base64, secrets, json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, session, send_from_directory

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# ==========================================
# 密碼驗證 (PBKDF2-SHA256)
# ==========================================
PASSWORD_FILE = os.path.join(os.path.dirname(__file__), 'password.txt')

def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 260000)

def verify_password(password: str) -> bool:
    try:
        with open(PASSWORD_FILE, 'r') as f:
            line = f.read().strip()
        algo, iters, encoded = line.split('$')
        assert algo == 'pbkdf2_sha256'
        raw = base64.b64decode(encoded)
        salt, stored_key = raw[:16], raw[16:]
        return secrets.compare_digest(stored_key, _hash_password(password, salt))
    except Exception:
        return False

def set_password(new_password: str):
    salt = os.urandom(16)
    key = _hash_password(new_password, salt)
    encoded = base64.b64encode(salt + key).decode()
    with open(PASSWORD_FILE, 'w') as f:
        f.write(f'pbkdf2_sha256$260000${encoded}\n')

# ==========================================
# 捷運路線與車站設定
# ==========================================
MRT_LINES = {
    'BL': {'name': '板南線',     'color': '#0070BD', 'cars': 6, 'directions': ['頂埔', '南港展覽館']},
    'R':  {'name': '淡水信義線', 'color': '#E3002C', 'cars': 6, 'directions': ['淡水', '象山']},
    'G':  {'name': '松山新店線', 'color': '#008659', 'cars': 6, 'directions': ['新店', '松山']},
    'O':  {'name': '中和新蘆線', 'color': '#F8A800', 'cars': 6, 'directions': ['迴龍/蘆洲', '南勢角']},
    'BR': {'name': '文湖線',     'color': '#C48A00', 'cars': 4, 'directions': ['動物園', '南港展覽館']},
    'Y':  {'name': '環狀線',     'color': '#FFDB00', 'cars': 4, 'directions': ['大坪林', '新北產業園區']},
}

STATION_CONFIG = {
    'station_name': '忠孝復興',
    'line_id': 'BL',
    'cars': 6,
}

# ==========================================
# 系統狀態
# ==========================================
DEFAULT_MODEL_PATH = r"/home/inf436/head_detection_train/MRT_appv3.5/yolov12best.pt"

_model  = None
_device = 'cpu'

DIRECTIONS = ('A', 'B')

def _make_cars(n):
    return [{'id': i+1, 'count': 0, 'cap': 30} for i in range(n)]

SYSTEM_STATE = {
    'modelReady': False,
    'conf': 0.5,
    'iou': 0.45,
    'cars_A': _make_cars(STATION_CONFIG['cars']),
    'cars_B': _make_cars(STATION_CONFIG['cars']),
}

CAMERAS: dict = {}

# ==========================================
# 鏡頭設定持久化 (cameras.json)
# ==========================================
CAMERAS_FILE = os.path.join(os.path.dirname(__file__), 'cameras.json')

def _save_cameras():
    """將目前所有鏡頭設定寫入 cameras.json"""
    data = []
    for (direction, car_id), cam in CAMERAS.items():
        data.append({
            'direction': direction,
            'car_id':    car_id,
            'src_type':  cam.src_type,
            'src':       cam.src,
        })
    with open(CAMERAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_cameras():
    """啟動時從 cameras.json 還原鏡頭（不自動啟動，需手動 start 或由此處啟動）"""
    if not os.path.exists(CAMERAS_FILE):
        return
    try:
        with open(CAMERAS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:
            direction = entry['direction'].upper()
            car_id    = int(entry['car_id'])
            src_type  = entry['src_type']
            src       = entry['src']
            if direction not in DIRECTIONS:
                continue
            key = (direction, car_id)
            cam = CameraThread(direction, car_id, src_type, src)
            CAMERAS[key] = cam
            cam.start()
        print(f"[cameras.json] 已還原 {len(data)} 個鏡頭設定")
    except Exception as e:
        print(f"[cameras.json] 讀取失敗: {e}")

# ==========================================
# 單一推論佇列 — 所有鏡頭排隊，一個 worker 執行緒依序推論
# ==========================================
import queue

_infer_queue: queue.Queue = queue.Queue(maxsize=12)

def _infer_worker():
    while True:
        cam, img = _infer_queue.get()
        try:
            if not cam.active:
                continue
            t0 = time.perf_counter()
            results = _model.predict(
                img,
                device=_device,
                conf=SYSTEM_STATE['conf'],
                iou=SYSTEM_STATE['iou'],
                imgsz=640,
                verbose=False,
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            count = 0
            for r in results:
                for box in r.boxes:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    c = float(box.conf[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (246, 130, 59), 2)
                    cv2.putText(img, f"{int(c*100)}%", (x1, max(10, y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (246, 130, 59), 2)
            ret, jpeg = cv2.imencode('.jpg', img)
            if ret:
                cam.current_frame = jpeg.tobytes()
            cam.count   = count
            cam.latency = latency_ms
            cars = SYSTEM_STATE[f'cars_{cam.direction}']
            idx  = cam.car_id - 1
            if 0 <= idx < len(cars):
                cars[idx]['count'] = count
        except Exception as e:
            print(f"推論錯誤 [{cam.direction}-C{cam.car_id}]: {e}")
        finally:
            _infer_queue.task_done()

threading.Thread(target=_infer_worker, daemon=True).start()

# ==========================================
# 網路串流請求佇列 — HTTP/RTSP 鏡頭每次只有一個發出請求
# 避免同時大量連線被伺服器擋掉
# ==========================================
_fetch_queue: queue.Queue = queue.Queue()

def _fetch_worker():
    """
    依序從佇列取出網路串流鏡頭，執行一次 cap.read()，
    完成後才處理下一個，避免同時大量連線。
    """
    while True:
        cam = _fetch_queue.get()
        try:
            if not cam.active:
                continue
            cap = cv2.VideoCapture(cam.src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 只保留最新一幀
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                cam._enqueue(frame)
                # 即使模型未載入，也先存原始 frame 以供預覽
                if _model is None:
                    ret2, jpeg = cv2.imencode('.jpg', frame)
                    if ret2:
                        cam.current_frame = jpeg.tobytes()
        except Exception as e:
            print(f"串流擷取錯誤 [{cam.direction}-C{cam.car_id}]: {e}")
        finally:
            _fetch_queue.task_done()

threading.Thread(target=_fetch_worker, daemon=True).start()

def get_device():
    try:
        import torch
        if torch.cuda.is_available(): return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
    except: pass
    return 'cpu'

# ==========================================
# 背景擷取執行緒（每個鏡頭一條）
# ==========================================
class CameraThread:
    INFER_INTERVAL = 5.0  # 每幾秒送一張 frame 去推論

    def __init__(self, direction: str, car_id: int, src_type: str, src: str):
        self.direction     = direction
        self.car_id        = car_id
        self.src_type      = src_type
        self.src           = src
        self.active        = False
        self.current_frame = None
        self.thread        = None
        self.count         = 0
        self.latency       = 0

    def start(self):
        self.active = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def stop(self):
        self.active = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _enqueue(self, img):
        if _model is None or img is None:
            return
        try:
            _infer_queue.put_nowait((self, img.copy()))
        except queue.Full:
            pass

    def update(self):
        if self.src_type == 'folder':
            # ── 資料夾模式：每 INFER_INTERVAL 秒換一張圖 ──
            while self.active:
                images = []
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
                    images.extend(glob.glob(os.path.join(self.src, ext)))
                    images.extend(glob.glob(os.path.join(self.src, ext.upper())))
                images = sorted(list(set(images)))
                if not images:
                    time.sleep(2)
                    continue
                for img_path in images:
                    if not self.active:
                        break
                    img = cv2.imread(img_path)
                    self._enqueue(img)
                    time.sleep(self.INFER_INTERVAL)

        elif self.src_type == 'webcam':
            # ── 本機攝影機：持續讀取，每 INFER_INTERVAL 秒推論一次 ──
            src_val  = int(self.src) if self.src.isdigit() else self.src
            cap      = cv2.VideoCapture(src_val)
            last_inf = 0.0
            while self.active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                # 即時預覽（不等推論）
                ret2, jpeg = cv2.imencode('.jpg', frame)
                if ret2:
                    self.current_frame = jpeg.tobytes()
                now = time.perf_counter()
                if now - last_inf >= self.INFER_INTERVAL:
                    self._enqueue(frame)
                    last_inf = now
                else:
                    time.sleep(0.05)
            if cap:
                cap.release()

        else:
            # ── 網路串流 (HTTP/RTSP)：透過佇列每 INFER_INTERVAL 秒發一次請求 ──
            while self.active:
                try:
                    _fetch_queue.put_nowait(self)
                except queue.Full:
                    pass  # 佇列滿了就跳過本輪，等下一輪
                time.sleep(self.INFER_INTERVAL)

# ==========================================
# 管理端身份驗證 Decorator
# ==========================================
from functools import wraps

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return jsonify({'error': 'unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/tabicon.ico')
def favicon():
    return send_from_directory('static', 'tabicon.ico', mimetype='image/x-icon')

# ==========================================
# 公開路由
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/public/station')
def api_public_station():
    line = MRT_LINES.get(STATION_CONFIG['line_id'], {})
    return jsonify({
        'station':    STATION_CONFIG['station_name'],
        'line_id':    STATION_CONFIG['line_id'],
        'line_name':  line.get('name', ''),
        'line_color': line.get('color', '#888'),
        'cars':       STATION_CONFIG['cars'],
        'directions': line.get('directions', ['方向A', '方向B']),
    })

@app.route('/api/public/crowd')
def api_public_crowd():
    return jsonify({
        'modelReady': SYSTEM_STATE['modelReady'],
        'cars_A':     SYSTEM_STATE['cars_A'],
        'cars_B':     SYSTEM_STATE['cars_B'],
    })

# ==========================================
# 管理員登入/登出
# ==========================================
@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    data = request.json or {}
    if verify_password(data.get('password', '')):
        session['admin_logged_in'] = True
        return jsonify({'ok': True})
    return jsonify({'error': '密碼錯誤'}), 403

@app.route('/api/admin/logout', methods=['POST'])
def api_admin_logout():
    session.pop('admin_logged_in', None)
    return jsonify({'ok': True})

@app.route('/api/admin/check')
def api_admin_check():
    return jsonify({'ok': session.get('admin_logged_in', False)})

# ==========================================
# 管理員 API
# ==========================================
@app.route('/api/admin/lines')
@admin_required
def api_lines():
    return jsonify(MRT_LINES)

@app.route('/api/admin/station_config', methods=['GET'])
@admin_required
def api_get_station_config():
    return jsonify(STATION_CONFIG)

@app.route('/api/admin/station_config', methods=['POST'])
@admin_required
def api_set_station_config():
    data = request.json or {}
    if 'station_name' in data:
        STATION_CONFIG['station_name'] = str(data['station_name'])
    if 'line_id' in data and data['line_id'] in MRT_LINES:
        STATION_CONFIG['line_id'] = data['line_id']
        STATION_CONFIG['cars']    = MRT_LINES[data['line_id']]['cars']
    if 'cars' in data:
        n = int(data['cars'])
        STATION_CONFIG['cars'] = max(2, min(10, n))
        for dk in ('cars_A', 'cars_B'):
            old = {c['id']: c['count'] for c in SYSTEM_STATE[dk]}
            SYSTEM_STATE[dk] = _make_cars(STATION_CONFIG['cars'])
            for c in SYSTEM_STATE[dk]:
                c['count'] = old.get(c['id'], 0)
    return jsonify({'ok': True, 'config': STATION_CONFIG})

@app.route('/api/admin/change_password', methods=['POST'])
@admin_required
def api_change_password():
    data   = request.json or {}
    old_pw = data.get('old_password', '')
    new_pw = data.get('new_password', '')
    if not verify_password(old_pw):
        return jsonify({'error': '舊密碼錯誤'}), 403
    if len(new_pw) < 6:
        return jsonify({'error': '新密碼至少需要 6 個字元'}), 400
    set_password(new_pw)
    return jsonify({'ok': True})

@app.route('/api/admin/load', methods=['POST'])
@admin_required
def api_load():
    global _model, _device
    try:
        from ultralytics import YOLO
        _device = get_device()
        _model  = YOLO(DEFAULT_MODEL_PATH)
        _model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=_device, verbose=False)
        SYSTEM_STATE['modelReady'] = True
        return jsonify({'ok': True, 'device': _device})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/state')
@admin_required
def api_state():
    cams_info = []
    for (direction, car_id), cam in CAMERAS.items():
        cams_info.append({
            'direction': direction,
            'carId':     car_id,
            'type':      cam.src_type,
            'src':       cam.src,
            'online':    cam.active,
            'count':     cam.count,
            'latency':   cam.latency,
        })
    return jsonify({
        'modelReady': SYSTEM_STATE['modelReady'],
        'conf':       SYSTEM_STATE['conf'],
        'iou':        SYSTEM_STATE['iou'],
        'cars_A':     SYSTEM_STATE['cars_A'],
        'cars_B':     SYSTEM_STATE['cars_B'],
        'cameras':    cams_info,
    })

@app.route('/api/admin/settings', methods=['POST'])
@admin_required
def api_settings():
    data = request.json or {}
    if 'conf' in data: SYSTEM_STATE['conf'] = float(data['conf'])
    if 'iou'  in data: SYSTEM_STATE['iou']  = float(data['iou'])
    return jsonify({'ok': True})

@app.route('/api/admin/camera', methods=['POST'])
@admin_required
def api_camera():
    data      = request.json or {}
    action    = data.get('action')
    direction = data.get('direction', 'A').upper()
    car_id    = int(data.get('carId'))
    key       = (direction, car_id)

    if direction not in DIRECTIONS:
        return jsonify({'error': '無效方向'}), 400

    state_key = f'cars_{direction}'

    if action == 'add':
        if key in CAMERAS:
            CAMERAS[key].stop()
        cam = CameraThread(direction, car_id, data.get('type'), data.get('src'))
        CAMERAS[key] = cam
        cam.start()
        _save_cameras()  # ← 新增後儲存

    elif action == 'remove':
        if key in CAMERAS:
            CAMERAS[key].stop()
            del CAMERAS[key]
        cars = SYSTEM_STATE[state_key]
        if 0 <= car_id - 1 < len(cars):
            cars[car_id - 1]['count'] = 0
        _save_cameras()  # ← 刪除後儲存

    elif action == 'toggle':
        if key in CAMERAS:
            if data.get('online'):
                CAMERAS[key].start()
            else:
                CAMERAS[key].stop()
                cars = SYSTEM_STATE[state_key]
                if 0 <= car_id - 1 < len(cars):
                    cars[car_id - 1]['count'] = 0
        # toggle 不影響儲存的設定（只是暫停/恢復）

    return jsonify({'ok': True})

@app.route('/api/admin/stream/<string:direction>/<int:car_id>')
@admin_required
def video_feed(direction: str, car_id: int):
    key = (direction.upper(), car_id)
    def generate():
        cam = CAMERAS.get(key)
        if not cam:
            return
        while getattr(cam, 'active', False):
            frame = cam.current_frame
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# 啟動時還原鏡頭設定
# ==========================================
if __name__ == '__main__':
    _load_cameras()  # ← 讀取 cameras.json 還原鏡頭
    app.run(debug=False, host='0.0.0.0', port=5005, threaded=True)