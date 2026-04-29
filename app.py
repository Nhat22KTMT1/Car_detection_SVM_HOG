import cv2, joblib, numpy as np, threading, time, os
from flask import Flask, Response, render_template_string
from scipy.ndimage import label as scipy_label

MODEL_PATH   = "vehicle_svm_v3.pkl"
WIN          = (64, 64)
LINE_RATIO   = 0.65
LINE_MARGIN  = 20
THRESHOLD    = 0.5
SCALES       = (1.0, 1.5)
STEP_CELLS   = 3
ROI_TOP      = 0.4
ROI_BOTTOM   = 0.92
CAP_W        = 320
CAP_H        = 240
JPEG_Q       = 60
DETECT_EVERY = 3
HEAT_DECAY   = 0.92
HEAT_THRESH  = 1.5
_HTML_INLINE = '''<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Vehicle Counter</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0 }
    body { background: #080c14; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif;
           min-height: 100vh; display: flex; flex-direction: column;
           align-items: center; padding: 28px 16px; gap: 20px }
    .title { font-size: 1.7rem; font-weight: 800;
             background: linear-gradient(90deg, #f97316, #ef4444);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent }
    .subtitle { font-size: .78rem; color: #475569; margin-top: 2px }
    .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 18px;
            padding: 16px; width: 100%; max-width: 720px;
            box-shadow: 0 30px 60px rgba(0,0,0,.7) }
    .stream-wrap { position: relative; border-radius: 12px; overflow: hidden }
    .stream-wrap img { width: 100%; display: block; border-radius: 12px }
    .overlay-badge { position: absolute; top: 10px; right: 12px;
                     background: rgba(0,0,0,.65); border: 1px solid rgba(255,255,255,.12);
                     border-radius: 8px; padding: 5px 12px; font-size: .8rem;
                     color: #94a3b8; display: flex; gap: 14px }
    .overlay-badge b { color: #f97316 }
    .stats { display: flex; gap: 12px; margin-top: 14px;
             justify-content: center; flex-wrap: wrap }
    .badge { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
             padding: 10px 22px; text-align: center; min-width: 100px }
    .badge .val { font-size: 1.6rem; font-weight: 700; color: #f97316; line-height: 1 }
    .badge .lbl { font-size: .72rem; color: #64748b; margin-top: 4px }
    .controls { display: flex; gap: 10px; margin-top: 14px; justify-content: center }
    button { padding: 9px 24px; border: none; border-radius: 9px; font-size: .88rem;
             font-weight: 600; cursor: pointer; transition: opacity .2s, transform .1s }
    button:hover  { opacity: .85 }
    button:active { transform: scale(.97) }
    .btn-r { background: #ef4444; color: #fff }
    .info { font-size: .72rem; color: #334155; text-align: center; margin-top: 10px }
  </style>
</head>
<body>
  <div>
    <div class="title">🚗 Vehicle Counter</div>
    <div class="subtitle">HoG + LinearSVM · Heatmap · Threaded · Realtime</div>
  </div>
  <div class="card">
    <div class="stream-wrap">
      <img id="stream" src="/video">
      <div class="overlay-badge">
        <span>Stream <b id="fps_s">—</b> FPS</span>
        <span>Detect <b id="fps_d">—</b> FPS</span>
      </div>
    </div>
    <div class="stats">
      <div class="badge"><div class="val" id="cnt">0</div><div class="lbl">Xe đã đếm</div></div>
      <div class="badge"><div class="val" id="fps_sv">—</div><div class="lbl">Stream FPS</div></div>
      <div class="badge"><div class="val" id="fps_dv">—</div><div class="lbl">Detect FPS</div></div>
    </div>
    <div class="controls">
      <button class="btn-r" onclick="doReset()">🔄 Reset</button>
    </div>
    <p class="info">Detect thread chạy song song — stream không bị chặn bởi HoG+SVM</p>
  </div>
  <script>
    let t0=Date.now(), n=0;
    document.getElementById('stream').onload = () => {
      n++;
      if (n % 10 === 0) {
        const fps = (10000/(Date.now()-t0)).toFixed(1);
        document.getElementById('fps_s').textContent = fps;
        document.getElementById('fps_sv').textContent = fps;
        t0 = Date.now();
      }
    };
    setInterval(() => {
      fetch('/stats').then(r => r.json()).then(d => {
        document.getElementById('cnt').textContent    = d.count;
        document.getElementById('fps_d').textContent  = d.fps_detect;
        document.getElementById('fps_dv').textContent = d.fps_detect;
      });
    }, 500);
    function doReset() { fetch('/reset') }
  </script>
</body>
</html>'''

model_full = joblib.load(MODEL_PATH)
scaler     = model_full.named_steps["scaler"]
clf        = model_full.named_steps["clf"]

HOG_CV = cv2.HOGDescriptor(
    _winSize=(64,64), _blockSize=(16,16),
    _blockStride=(8,8), _cellSize=(8,8), _nbins=9
)

class Heatmap:
    def __init__(self, shape):
        self.map = np.zeros(shape[:2], dtype=np.float32)
    def update(self, boxes):
        self.map *= HEAT_DECAY
        for b in boxes:
            self.map[b[1]:b[3], b[0]:b[2]] += 1.5
    def get_boxes(self):
        binary = (self.map >= HEAT_THRESH).astype(np.uint8)
        labeled, n = scipy_label(binary)
        boxes = []
        for k in range(1, n+1):
            nz = (labeled==k).nonzero()
            if len(nz[0]) < 150: continue
            y1,y2 = int(nz[0].min()),int(nz[0].max())
            x1,x2 = int(nz[1].min()),int(nz[1].max())
            boxes.append((x1,y1,x2,y2))
        return boxes

def nms(boxes, scores,iou_thresh =0.4):
    if not boxes: return []
    b=np.array(boxes,dtype=np.float32); s=np.array(scores)
    x1,y1,x2,y2=b[:,0],b[:,1],b[:,2],b[:,3]
    areas=(x2-x1+1)*(y2-y1+1); order=s.argsort()[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1+1)*np.maximum(0,yy2-yy1+1)
        iou=inter/(areas[i]+areas[order[1:]]-inter)
        order=order[np.where(iou<=thr)[0]+1]
    return keep

def detect(frame):
    H,W = frame.shape[:2]
    y1o,y2o = int(H*ROI_TOP),int(H*ROI_BOTTOM)
    roi = frame[y1o:y2o]
    all_boxes, all_scores = [], []
    for scale in SCALES:
        rh,rw = roi.shape[:2]
        nw,nh = int(rw/scale),int(rh/scale)
        if nw<WIN[0] or nh<WIN[1]: continue
        sroi = cv2.resize(roi,(nw,nh))
        ncx = nw//8; ncy = nh//8; cwc = WIN[0]//8
        for fy in range(0, ncy-cwc, STEP_CELLS):
            for fx in range(0, ncx-cwc, STEP_CELLS):
                px,py = fx*8, fy*8
                patch = sroi[py:py+WIN[1], px:px+WIN[0]]
                if patch.shape[:2]!=(WIN[1],WIN[0]): continue
                feat = scaler.transform(HOG_CV.compute(patch).flatten().reshape(1,-1))
                p = clf.predict_proba(feat)[0][1]
                if p >= THRESHOLD:
                    all_boxes.append([int(px*scale),int(py*scale)+y1o,
                                      int((px+WIN[0])*scale),int((py+WIN[1])*scale)+y1o])
                    all_scores.append(p)
    keep = nms(all_boxes, all_scores)
    return [all_boxes[i] for i in keep]

state = {
    "raw_frame"  : None,
    "annot_frame": None,
    "count"      : 0,
    "fps_detect" : 0.0,
    "fps_stream" : 0.0,
    "lock_raw"   : threading.Lock(),
    "lock_annot" : threading.Lock(),
}

def camera_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FPS,          30)
    while True:
        ret, frame = cap.read()
        if not ret: continue
        with state["lock_raw"]:
            state["raw_frame"] = frame

def detect_thread():
    heat         = None
    vehicle_count= 0
    counted_positions = []   # lưu tọa độ X của xe vừa được đếm
    COOLDOWN_FRAMES   = 8    # sau khi đếm, khóa vùng đó trong N frame
    cooldown_counters = {}   # {x_position: frames_remaining}
    prev_centers = []
    frame_idx    = 0
    COLORS       = [(0,255,0),(255,0,255),(0,165,255),(255,255,0),(0,255,255)]
    t0           = time.time()
    while True:
        with state["lock_raw"]:
            frame = state["raw_frame"]
        if frame is None:
            time.sleep(0.005); continue
        H,W    = frame.shape[:2]
        LINE_Y = int(H * LINE_RATIO)
        if heat is None:
            heat = Heatmap((H,W))
        vis = frame.copy()
        if frame_idx % DETECT_EVERY == 0:
            raw_boxes = detect(frame)
            heat.update(raw_boxes)
        final_boxes = heat.get_boxes()
        #curr_c = [((b[0]+b[2])//2,(b[1]+b[3])//2) for b in final_boxes]
        #for cx,cy in curr_c:
        #    if abs(cy-LINE_Y) < LINE_MARGIN:
        #        if not any(abs(cx-p[0])<60 and abs(cy-p[1])<60 for p in prev_centers):
        #            vehicle_count += 1
        #prev_centers = curr_c
        # ── THAY BẰNG logic crossing chuẩn (giống run_on_video) ─────
        curr_c = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in final_boxes]

        for cx, cy in curr_c:
            min_dist   = float('inf')
            matched_py = None
            max_dist   = max(100, H * 0.15)

            for px, py in prev_centers:
                dist = ((cx-px)**2 + (cy-py)**2)**0.5
                if dist < max_dist and dist < min_dist:
                    min_dist   = dist
                    matched_py = py

            if matched_py is not None:
                if matched_py < LINE_Y and cy >= LINE_Y:
                    vehicle_count += 1

        prev_centers = curr_c
        state["count"] = vehicle_count
        for i,(x1,y1,x2,y2) in enumerate(final_boxes):
            col = COLORS[i % len(COLORS)]
            cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
            cv2.circle(vis,((x1+x2)//2,(y1+y2)//2),5,col,-1)
        cv2.line(vis,(0,LINE_Y),(W,LINE_Y),(0,0,255),2)
        frame_idx += 1
        if frame_idx % 10 == 0:
            state["fps_detect"] = round(10/(time.time()-t0), 1)
            t0 = time.time()
        ov = vis.copy()
        cv2.rectangle(ov,(0,0),(320,80),(0,0,0),-1)
        cv2.addWeighted(ov,0.4,vis,0.6,0,vis)
        cv2.putText(vis, "Count: " + str(vehicle_count),
                    (10,46),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        fps_txt = "det=" + str(state["fps_detect"]) + " str=" + str(state["fps_stream"])
        cv2.putText(vis, "FPS " + fps_txt,
                    (10,68),cv2.FONT_HERSHEY_SIMPLEX,0.52,(180,180,180),1)
        with state["lock_annot"]:
            state["annot_frame"] = vis

def gen_frames():
    t0 = time.time(); n = 0
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]
    while True:
        with state["lock_annot"]:
            frame = state["annot_frame"]
        if frame is None:
            time.sleep(0.01); continue
        _, buf = cv2.imencode(".jpg", frame, enc_params)
        n += 1
        if n % 15 == 0:
            state["fps_stream"] = round(15/(time.time()-t0), 1)
            t0 = time.time()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

app = Flask(__name__)

HTML = open("templates/index.html").read() if os.path.exists("templates/index.html") else _HTML_INLINE

@app.route("/")
def index(): return render_template_string(_HTML_INLINE)

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    return {"count": state["count"], "fps_detect": state["fps_detect"], "fps_stream": state["fps_stream"]}

@app.route("/reset")
def reset():
    state["count"] = 0
    return {"ok": True}

if __name__ == "__main__":
    print("Starting threads...")
    threading.Thread(target=camera_thread, daemon=True).start()
    time.sleep(0.5)
    threading.Thread(target=detect_thread, daemon=True).start()
    print("Server: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
