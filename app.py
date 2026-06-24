from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import cv2
import base64
import numpy as np
import os

app = Flask(__name__)

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.pt"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "log_canh_bao.txt"

UPLOAD_CONF = float(os.getenv("UPLOAD_CONF", "0.10"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB


# =========================
# INIT LOG
# =========================
if not LOG_FILE.exists():
    LOG_FILE.write_text(
        f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] SYSTEM INIT: Hệ thống sẵn sàng.\n",
        encoding="utf-8"
    )


# =========================
# LOAD MODEL
# =========================
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    print("✅ Đã load Model AI thành công!")
    print("✅ Model path:", MODEL_PATH)
    print("✅ Model classes:", model.names)

except Exception as e:
    print("❌ Lỗi load model:", e)
    model = None


# =========================
# HELPER
# =========================
def ghi_log(noi_dung):
    thoi_gian = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{thoi_gian}] {noi_dung}\n")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def lay_ten_class(class_id):
    """
    Model hiện tại đang hiển thị class dạng 0/1 nên để chung KHÓI/LỬA
    để tránh bị nhầm khói thành lửa hoặc ngược lại.
    """
    return "KHÓI/LỬA"


def phan_tich_ket_qua(result):
    danh_sach = []

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return danh_sach

    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        danh_sach.append({
            "ten": lay_ten_class(class_id),
            "conf": conf,
            "xyxy": (x1, y1, x2, y2)
        })

    return danh_sach


def tao_thong_bao_ket_qua(danh_sach):
    if not danh_sach:
        return "✅ Không phát hiện khói/lửa."

    conf_cao_nhat = max(item["conf"] for item in danh_sach)
    return f"🚨 Phát hiện khói/lửa ({conf_cao_nhat:.2f})"


def ve_ket_qua(anh_cv2, danh_sach):
    anh_out = anh_cv2.copy()

    for item in danh_sach:
        x1, y1, x2, y2 = item["xyxy"]
        conf = item["conf"]

        label = f"PHAT HIEN KHOI/LUA {conf:.2f}"

        mau_do = (0, 0, 255)
        mau_trang = (255, 255, 255)

        cv2.rectangle(anh_out, (x1, y1), (x2, y2), mau_do, 3)

        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        y_label = max(0, y1 - text_h - 12)

        cv2.rectangle(
            anh_out,
            (x1, y_label),
            (x1 + text_w + 10, y1),
            mau_do,
            -1
        )

        cv2.putText(
            anh_out,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mau_trang,
            2
        )

    return anh_out


def ma_hoa_anh_base64(anh_cv2):
    ret, buffer = cv2.imencode(".png", anh_cv2)
    if not ret:
        raise ValueError("Không thể mã hóa ảnh kết quả.")
    return base64.b64encode(buffer).decode("utf-8")


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template(
        "index.html",
        anh_ket_qua_b64=None,
        ket_qua_text=None,
        phat_hien=False,
        ket_qua=[],
        loi=None
    )


@app.route("/upload_anh", methods=["POST"])
def upload_anh():
    try:
        if model is None:
            return render_template(
                "index.html",
                loi="Model chưa load được. Kiểm tra file models/best.pt.",
                anh_ket_qua_b64=None,
                ket_qua_text=None,
                phat_hien=False,
                ket_qua=[]
            )

        file = request.files.get("file_anh")

        if file is None or file.filename == "":
            return render_template(
                "index.html",
                loi="Bạn chưa chọn ảnh.",
                anh_ket_qua_b64=None,
                ket_qua_text=None,
                phat_hien=False,
                ket_qua=[]
            )

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                loi="Chỉ hỗ trợ ảnh .jpg, .jpeg, .png, .webp.",
                anh_ket_qua_b64=None,
                ket_qua_text=None,
                phat_hien=False,
                ket_qua=[]
            )

        filename = secure_filename(file.filename)

        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        anh_cv2 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if anh_cv2 is None:
            return render_template(
                "index.html",
                loi="File ảnh bị lỗi hoặc không hợp lệ.",
                anh_ket_qua_b64=None,
                ket_qua_text=None,
                phat_hien=False,
                ket_qua=[]
            )

        result = model.predict(
            anh_cv2,
            conf=UPLOAD_CONF,
            imgsz=IMG_SIZE,
            verbose=False
        )[0]

        ket_qua = phan_tich_ket_qua(result)
        ket_qua_text = tao_thong_bao_ket_qua(ket_qua)
        phat_hien = len(ket_qua) > 0

        if phat_hien:
            ghi_log(f"🚨 PHÂN TÍCH ẢNH: {filename} - {ket_qua_text}")
        else:
            ghi_log(f"✅ PHÂN TÍCH ẢNH: {filename} - Không phát hiện khói/lửa.")

        anh_out = ve_ket_qua(anh_cv2, ket_qua)
        anh_b64 = ma_hoa_anh_base64(anh_out)

        return render_template(
            "index.html",
            anh_ket_qua_b64=anh_b64,
            ket_qua_text=ket_qua_text,
            phat_hien=phat_hien,
            ket_qua=ket_qua,
            loi=None
        )

    except Exception as e:
        print("❌ Lỗi xử lý ảnh:", e)
        return render_template(
            "index.html",
            loi=f"Lỗi xử lý ảnh: {e}",
            anh_ket_qua_b64=None,
            ket_qua_text=None,
            phat_hien=False,
            ket_qua=[]
        )


@app.route("/lay_log")
def lay_log():
    try:
        if not LOG_FILE.exists():
            return jsonify([])

        logs = LOG_FILE.read_text(encoding="utf-8").splitlines()
        return jsonify(logs[-12:])

    except Exception:
        return jsonify([])


@app.route("/xoa_log", methods=["POST"])
def xoa_log():
    LOG_FILE.write_text(
        f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] SYSTEM: Đã xóa lịch sử hệ thống.\n",
        encoding="utf-8"
    )
    return jsonify({"status": "ok"})


@app.errorhandler(413)
def file_qua_lon(e):
    return render_template(
        "index.html",
        loi="File quá lớn. Chỉ cho phép tối đa 10MB.",
        anh_ket_qua_b64=None,
        ket_qua_text=None,
        phat_hien=False,
        ket_qua=[]
    ), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)