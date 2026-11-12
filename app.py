import os
import math
import threading

import cv2
import numpy as np
import streamlit as st
import av

from streamlit_webrtc import webrtc_streamer
from streamlit_autorefresh import st_autorefresh


# ==========================================
# 1. ページ設定
# ==========================================

st.set_page_config(
    page_title="リアルタイム播種量モニター",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 リアルタイム播種量モニター")
st.write("カメラ映像から種もみの被覆面積を測定し、重なりを補正して播種量を推定します。")


# ==========================================
# 2. スレッド間共有データ
# ==========================================

class SharedData:

    def __init__(self):

        self.lock = threading.Lock()

        self.settings = {}

        self.visible_coverage = 0.0
        self.corrected_coverage = 0.0
        self.estimated_weight = 0.0

        self.error = ""


if "shared_data" not in st.session_state:
    st.session_state["shared_data"] = SharedData()

shared = st.session_state["shared_data"]


# ==========================================
# 3. 実測重量による校正
# ==========================================

if "weight_coefficient" not in st.session_state:
    st.session_state["weight_coefficient"] = 5.0

if "measured_weight" not in st.session_state:
    st.session_state["measured_weight"] = 0.0


def apply_calibration():

    measured = float(
        st.session_state["measured_weight"]
    )

    with shared.lock:
        coverage = shared.corrected_coverage

    if coverage <= 0 or measured <= 0:

        st.session_state["calibration_status"] = (
            "計測中に正の実測重量を入力してください。"
        )

    else:

        new_coefficient = measured / coverage

        st.session_state["weight_coefficient"] = (
            new_coefficient
        )

        st.session_state["calibration_status"] = (
            f"校正完了：1%あたり "
            f"{new_coefficient:.3f} g に更新しました。"
        )


# ==========================================
# 4. サイドバー設定
# ==========================================

st.sidebar.header("⚙️ システム設定")


# --- 播種量設定 ---

st.sidebar.subheader("📐 播種量推定")

weight_coefficient = st.sidebar.number_input(
    "補正後1%あたりの重量 (g)",
    min_value=0.01,
    max_value=1000.0,
    step=0.1,
    key="weight_coefficient"
)

overlap_factor = st.sidebar.slider(
    "重なり補正係数",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0：補正なし、1：ランダムな重なりを仮定した補正を最大限適用"
)


# --- 実測重量による校正 ---

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ 実測値による校正")

st.sidebar.number_input(
    "実際に量った種もみの重量 (g)",
    min_value=0.0,
    step=0.1,
    key="measured_weight"
)

st.sidebar.button(
    "実測重量から係数を更新",
    on_click=apply_calibration
)

if st.session_state.get("calibration_status"):
    st.sidebar.info(
        st.session_state["calibration_status"]
    )


# --- HSV色抽出 ---

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 色抽出設定")

h_min = st.sidebar.slider(
    "色相 H 最小",
    0, 179, 15
)

h_max = st.sidebar.slider(
    "色相 H 最大",
    0, 179, 45
)

s_min = st.sidebar.slider(
    "彩度 S 最小",
    0, 255, 50
)

s_max = st.sidebar.slider(
    "彩度 S 最大",
    0, 255, 255
)

v_min = st.sidebar.slider(
    "明度 V 最小",
    0, 255, 50
)

v_max = st.sidebar.slider(
    "明度 V 最大",
    0, 255, 255
)


# --- 計測範囲 ---

st.sidebar.markdown("---")
st.sidebar.subheader("📏 計測範囲")

roi_x = st.sidebar.slider(
    "左右方向の範囲 (%)",
    0, 100, (0, 100)
)

roi_y = st.sidebar.slider(
    "上下方向の範囲 (%)",
    0, 100, (0, 100)
)


# --- ノイズ除去 ---

noise_kernel = st.sidebar.select_slider(
    "ノイズ除去の強さ",
    options=[1, 3, 5, 7],
    value=3
)


# ==========================================
# 5. 設定値を共有データに保存
# ==========================================

lower_color = np.array([
    min(h_min, h_max),
    min(s_min, s_max),
    min(v_min, v_max)
], dtype=np.uint8)

upper_color = np.array([
    max(h_min, h_max),
    max(s_min, s_max),
    max(v_min, v_max)
], dtype=np.uint8)

settings = {
    "lower": lower_color,
    "upper": upper_color,
    "roi_x": roi_x,
    "roi_y": roi_y,
    "noise_kernel": noise_kernel,
    "overlap_factor": overlap_factor,
    "weight_coefficient": float(weight_coefficient)
}

with shared.lock:
    shared.settings = settings


# ==========================================
# 6. 重なり補正関数
# ==========================================

def correct_coverage(coverage_percent, factor):

    # 被覆率を0～1に変換
    p = coverage_percent / 100.0

    # 100%付近での計算の発散を防止
    p = min(max(p, 0.0), 0.98)

    # ランダムな重なりを仮定した理論補正
    poisson_area = -math.log1p(-p)

    # 補正の強さを調整
    corrected = p + factor * (
        poisson_area - p
    )

    return corrected * 100.0


# ==========================================
# 7. リアルタイム映像処理
# ==========================================

def video_frame_callback(frame):

    try:

        # カメラ映像を取得
        img = frame.to_ndarray(
            format="bgr24"
        )

        # 処理負荷を軽減するため幅を最大640pxにする
        h, w = img.shape[:2]

        if w > 640:

            new_h = int(h * 640 / w)

            img = cv2.resize(
                img,
                (640, new_h)
            )

        h, w = img.shape[:2]

        # 最新の設定を取得
        with shared.lock:
            cfg = shared.settings.copy()

        # ==================================
        # 計測範囲の指定
        # ==================================

        x1 = int(w * cfg["roi_x"][0] / 100)
        x2 = int(w * cfg["roi_x"][1] / 100)

        y1 = int(h * cfg["roi_y"][0] / 100)
        y2 = int(h * cfg["roi_y"][1] / 100)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("計測範囲を確認してください。")

        roi = img[y1:y2, x1:x2]

        # ==================================
        # HSV変換
        # ==================================

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        # 種もみの色を抽出
        mask = cv2.inRange(
            hsv,
            cfg["lower"],
            cfg["upper"]
        )

        # ==================================
        # ノイズ除去
        # ==================================

        if cfg["noise_kernel"] > 1:

            kernel = np.ones(
                (
                    cfg["noise_kernel"],
                    cfg["noise_kernel"]
                ),
                np.uint8
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

        # ==================================
        # 被覆面積の計算
        # ==================================

        total_pixels = roi.shape[0] * roi.shape[1]

        seed_pixels = cv2.countNonZero(mask)

        coverage_ratio = (
            seed_pixels / total_pixels
        ) * 100.0

        # ==================================
        # 重なり補正
        # ==================================

        corrected_ratio = correct_coverage(
            coverage_ratio,
            cfg["overlap_factor"]
        )

        # ==================================
        # 推定重量計算
        # ==================================

        estimated_weight = (
            corrected_ratio *
            cfg["weight_coefficient"]
        )

        # ==================================
        # 結果表示用画像
        # ==================================

        # マスク画像を元画像と同じ大きさにする
        mask_full = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        mask_full[y1:y2, x1:x2] = mask

        # 元画像
        original = img.copy()

        cv2.rectangle(
            original,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 255),
            2
        )

        # マスク画像
        mask_display = cv2.cvtColor(
            mask_full,
            cv2.COLOR_GRAY2BGR
        )

        # 抽出結果に緑色を重ねる
        result = img.copy()

        selected = mask_full > 0

        green = np.array(
            [0, 255, 0],
            dtype=np.float32
        )

        result[selected] = (
            result[selected].astype(np.float32) * 0.5
            + green * 0.5
        ).astype(np.uint8)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 255),
            2
        )

        # ラベル表示
        cv2.putText(
            original,
            "Original",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            mask_display,
            "Mask",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            result,
            "Analysis",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            result,
            f"Coverage: {coverage_ratio:.2f}%",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            f"Corrected: {corrected_ratio:.2f}%",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # 3画面を横に並べる
        combined = cv2.hconcat([
            original,
            mask_display,
            result
        ])

        # ==================================
        # 計測値を共有
        # ==================================

        with shared.lock:

            shared.visible_coverage = coverage_ratio

            shared.corrected_coverage = corrected_ratio

            shared.estimated_weight = estimated_weight

            shared.error = ""

        # 映像をStreamlitに返す
        return av.VideoFrame.from_ndarray(
            combined,
            format="bgr24"
        )

    except Exception as e:

        with shared.lock:
            shared.error = str(e)

        return frame


# ==========================================
# 8. WebRTCカメラ接続
# ==========================================

st.markdown("---")
st.subheader("📸 リアルタイムカメラ")

st.write(
    "STARTを押すとカメラが起動し、連続して計測します。"
)


# Cloudflare TURNの認証情報が設定されていれば利用
rtc_args = {}

try:

    turn_id = str(
        st.secrets["CLOUDFLARE_TURN_KEY_ID"]
    )

    turn_token = str(
        st.secrets["CLOUDFLARE_TURN_KEY_API_TOKEN"]
    )

    if turn_id and turn_token:

        os.environ["CLOUDFLARE_TURN_KEY_ID"] = turn_id

        os.environ["CLOUDFLARE_TURN_KEY_API_TOKEN"] = turn_token

except Exception:

    # TURN未設定の場合はSTUNを使用
    rtc_args["rtc_configuration"] = {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }


ctx = webrtc_streamer(

    key="rice-seeding-monitor",

    video_frame_callback=video_frame_callback,

    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },

    **rtc_args

)


# ==========================================
# 9. リアルタイム計測結果
# ==========================================

st.markdown("---")
st.subheader("📊 現在の計測結果")

if ctx.state.playing:

    # カメラ起動中は0.5秒ごとに表示更新
    st_autorefresh(
        interval=500,
        key="live_refresh"
    )

with shared.lock:

    visible = shared.visible_coverage

    corrected = shared.corrected_coverage

    weight = shared.estimated_weight

    error = shared.error


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "実際に見えている被覆率",
        f"{visible:.2f}%"
    )

with col2:

    st.metric(
        "重なり補正後の換算被覆率",
        f"{corrected:.2f}%"
    )

with col3:

    st.metric(
        "推定播種量",
        f"{weight:.2f} g/枚"
    )


if not ctx.state.playing:

    st.info(
        "カメラを起動するとリアルタイム計測を開始します。"
    )

if error:

    st.error(f"画像処理エラー：{error}")


# ==========================================
# 10. 使用上の注意
# ==========================================

st.markdown("---")

st.subheader("📌 使用上の注意")

st.write("""
- トレイ全体が撮影範囲に入るようにしてください。
- 種もみの色に応じてHSV設定を調整してください。
- 背景と種もみの色が近いと誤検出が発生します。
- 重なり補正係数は実測重量と比較して調整してください。
- 実測重量による校正は、撮影対象と実測した試料を一致させて行ってください。
- 被覆率が非常に高い場合、面積だけで正確な重量を推定することは困難です。
""")
