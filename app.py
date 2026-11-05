import cv2
import numpy as np
import streamlit as st
import time

# ページ設定
st.set_page_config(page_title="リアルタイム播種量モニター", page_icon="🌾", layout="wide")
st.title("🌾 リアルタイム播種量モニター (動画ストリーム版)")
st.write("カメラの映像をリアルタイムで解析し、現在の播種量を常に画面に表示し続けます。")

# サイドバー：設定
st.sidebar.header("⚙️ 設定")
camera_id = st.sidebar.number_input("カメラID (内蔵: 0, USB: 1~)", min_value=0, max_value=5, value=0)
weight_coefficient = st.sidebar.number_input("1%あたりの播種量(g)目安", value=5.0)

st.sidebar.markdown("---")
st.sidebar.write("🎨 色抽出（キャリブレーション）")
h_min = st.sidebar.slider("色相 (H) 最小", 0, 179, 15)
h_max = st.sidebar.slider("色相 (H) 最大", 0, 179, 45)
s_min = st.sidebar.slider("彩度 (S) 最小", 0, 255, 50)
s_max = st.sidebar.slider("彩度 (S) 最大", 0, 255, 255)
v_min = st.sidebar.slider("明度 (V) 最小", 0, 255, 50)
v_max = st.sidebar.slider("明度 (V) 最大", 0, 255, 255)

lower_color = np.array([h_min, s_min, v_min])
upper_color = np.array([h_max, s_max, v_max])

# 起動・停止スイッチ
run_camera = st.checkbox("▶️ カメラを起動してリアルタイム監視を開始")

# 画面更新用のプレースホルダー（枠）を準備
st.markdown("### 📊 現在の計測値")
metric_placeholder = st.empty()

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("📸 元映像")
    frame_window = st.image([])
with col2:
    st.write("⚫️ 抽出マスク")
    mask_window = st.image([])
with col3:
    st.write("🟡 抽出結果")
    result_window = st.image([])

# 動画処理ループ
if run_camera:
    # カメラの初期化
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        st.error("エラー: カメラを読み込めません。カメラIDを変更するか、接続を確認してください。")
    
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("映像フレームの取得に失敗しました。")
            break
            
        # OpenCVのBGRをRGBに変換
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # HSV変換とマスク作成
        hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, lower_color, upper_color)
        extracted_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        
        # 面積計算
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        seed_pixels = cv2.countNonZero(mask)
        coverage_ratio = (seed_pixels / total_pixels) * 100
        estimated_weight = coverage_ratio * weight_coefficient
        
        # UIの更新（プレースホルダーに上書きし続ける）
        frame_window.image(img_rgb)
        mask_window.image(mask, cmap='gray')
        result_window.image(extracted_img)
        
        # 計測値の更新
        metric_placeholder.markdown(
            f"**種もみの被覆率:** `{coverage_ratio:.2f} %` ｜ "
            f"**推定播種量:** `{estimated_weight:.1f} g/枠`"
        )
        
        # 負荷軽減のためごくわずかにスリープ（必要に応じて調整）
        time.sleep(0.05)
        
    # チェックボックスが外されたらカメラを解放
    cap.release()
else:
    st.info("上の「▶️ カメラを起動」にチェックを入れると開始します。")
