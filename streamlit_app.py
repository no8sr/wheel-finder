import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Wheel Finder",
    page_icon="🚗",
    layout="wide"
)

st.title("ホイール検索アプリ")

st.write(
    "ホイールの画像とサイズ情報から、"
    "メーカー・モデルの候補を表示します。"
)

uploaded_file = st.file_uploader(
    "ホイール画像をアップロードしてください",
    type=["jpg", "jpeg", "png"]
)

st.subheader("検索条件")

col1, col2, col3 = st.columns(3)

with col1:
    inch = st.selectbox(
        "インチ",
        ["指定なし", 14, 15, 16, 17, 18, 19, 20, 21]
    )

with col2:
    holes = st.selectbox(
        "穴数",
        ["指定なし", 4, 5, 6]
    )

with col3:
    manufacturer = st.selectbox(
        "メーカー",
        [
            "指定なし",
            "メーカーA",
            "メーカーB",
            "メーカーC"
        ]
    )

color = st.selectbox(
    "色",
    [
        "指定なし",
        "シルバー",
        "ブラック",
        "ブロンズ",
        "ホワイト"
    ]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("入力画像")
        st.image(
            image,
            caption="アップロードされたホイール画像",
            width=400
        )

    except Exception:
        st.error("画像を正常に読み込めませんでした。")

if st.button("候補を検索", type="primary"):
    if uploaded_file is None:
        st.warning("ホイール画像をアップロードしてください。")

    else:
        st.subheader("検索結果")

        st.info(
            "現在は画面確認用の仮結果です。"
            "今後、CNNの予測結果に置き換えます。"
        )

        st.write("### 第1候補")
        st.write("メーカーA　モデルA")
        st.progress(0.86)
        st.write("画像の予測確率：86%")

        st.write("### 第2候補")
        st.write("メーカーB　モデルB")
        st.progress(0.72)
        st.write("画像の予測確率：72%")

        st.write("### 第3候補")
        st.write("メーカーC　モデルC")
        st.progress(0.61)
        st.write("画像の予測確率：61%")

st.divider()

st.caption(
    "検索結果は画像認識による候補であり、"
    "製品のメーカーおよびモデルを保証するものではありません。"
)