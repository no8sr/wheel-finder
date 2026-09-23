import streamlit as st
from PIL import Image

from predict import predict_image
DISPLAY_NAMES = {
    "BBS_LM": "BBS LM",
    "BBS_RI_A": "BBS RI-A",
    "RAYS_VR_G025": "RAYS VOLK RACING G025",
    "RAYS_VR_TE37_SAGA_Splus":
        "RAYS VOLK RACING TE37 SAGA S-plus",
    "WORK_EMOTION_CR_Kiwami":
        "WORK EMOTION CR Kiwami",
    "WORK_MEISTER_S1_3PIECE":
        "WORK MEISTER S1 3PIECE",
}

st.set_page_config(
    page_title="Wheel Finder",
    page_icon="🚗",
    layout="wide"
)


# アプリの説明
st.title("Wheel Finder")
st.subheader("画像からホイールのメーカー・モデル候補を検索")

st.write(
    "ホイールの画像をCNNで分析し、外観が近いメーカー・モデルを表示します。"
    "さらに、インチ、穴数、スポーク数などの情報を入力することで、"
    "候補を絞り込むことができます。"
)

st.info(
    "ホイール全体が写っている、正面または正面に近い画像を使用してください。"
)

with st.expander("使い方"):
    st.markdown(
        """
        1. ホイールの画像をアップロードします。
        2. 分かる範囲で検索条件を選択します。
        3. 「候補を検索」を押します。
        4. 検索結果として上位3件の候補が表示されます。

        分からない項目は「指定なし」のままで検索できます。
        """
    )

st.warning(
    "検索結果は画像認識による推定です。"
    "製品名や装着可否を保証するものではありません。"
)

st.divider()


# 画像のアップロード
st.header("1. ホイール画像の選択")

uploaded_file = st.file_uploader(
    "ホイール画像をアップロードしてください",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="アップロードされたホイール画像",
            width=400
        )

    except Exception:
        st.error("画像を正常に読み込めませんでした。")


# 検索条件
st.header("2. 検索条件")

st.write(
    "分かる情報だけ入力してください。"
    "不明な項目は「指定なし」のままで構いません。"
)

col1, col2, col3 = st.columns(3)

with col1:
    inch = st.selectbox(
        "インチ",
        ["指定なし", 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    )

with col2:
    holes = st.selectbox(
        "穴数",
        ["指定なし", 4, 5, 6]
    )

with col3:
    spoke_count = st.selectbox(
        "見かけ上のスポーク数",
        [
            "指定なし",
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            "13本以上",
            "数えにくい"
        ]
    )

col4, col5, col6 = st.columns(3)

with col4:
    spoke_type = st.selectbox(
        "スポーク構造",
        [
            "指定なし",
            "シングルスポーク",
            "ツインスポーク",
            "Y字スポーク",
            "V字スポーク",
            "メッシュ",
            "ディッシュ",
            "その他"
        ]
    )

with col5:
    color = st.selectbox(
        "色",
        [
            "指定なし",
            "シルバー",
            "ブラック",
            "ガンメタ",
            "ブロンズ",
            "ゴールド",
            "ホワイト",
            "その他"
        ]
    )

with col6:
    wheel_type = st.selectbox(
        "ホイールの種類",
        [
            "指定なし",
            "純正",
            "社外",
            "不明"
        ]
    )


# 詳細条件
with st.expander("詳細条件を入力する"):
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        manufacturer = st.selectbox(
            "メーカー",
            [
                "指定なし",
                "メーカーA",
                "メーカーB",
                "メーカーC"
            ]
        )

        pcd = st.selectbox(
            "PCD",
            [
                "指定なし",
                100,
                108,
                110,
                112,
                114.3,
                120,
                127,
                139.7
            ]
        )

        rim_width = st.selectbox(
            "リム幅（J数）",
            [
                "指定なし",
                4.5,
                5.0,
                5.5,
                6.0,
                6.5,
                7.0,
                7.5,
                8.0,
                8.5,
                9.0,
                9.5,
                10.0
            ]
        )

    with detail_col2:
        inset = st.number_input(
            "インセット",
            min_value=-100,
            max_value=100,
            value=0,
            step=1,
            help="不明な場合は入力せず、下のチェックを外してください。"
        )

        use_inset = st.checkbox(
            "インセットを検索条件に使用する"
        )

        pierce_bolt = st.selectbox(
            "ピアスボルト",
            [
                "指定なし",
                "あり",
                "なし"
            ]
        )

        construction = st.selectbox(
            "構造",
            [
                "指定なし",
                "1ピース",
                "2ピース",
                "3ピース"
            ]
        )

    engraving = st.text_input(
        "刻印・型番・センターキャップの文字",
        placeholder="例：WORK、RAYS、18×7.5J、ET45"
    )


# 検索実行
st.header("3. 検索")

if st.button(
    "候補を検索",
    type="primary",
    use_container_width=True
):
    if uploaded_file is None:
        st.warning("ホイール画像をアップロードしてください。")

    else:
        st.subheader("検索結果")

        results = predict_image(
    image,
    top_k=3
)

result_columns = st.columns(3)

for rank, result in enumerate(results):
    class_name = result["class_name"]
    probability = result["probability"]
    display_name = DISPLAY_NAMES.get(
        class_name,
        class_name
    )

    with result_columns[rank\]:
        st.write(f"### 第{rank + 1}候補")
        st.write(display_name)

        st.progress(
            float(probability)
        )

        st.write(
            f"画像の予測確率："
            f"{probability * 100:.1f}%"
        )


st.divider()

st.caption(
    "検索結果は画像認識と入力条件に基づく候補です。"
    "製品のメーカー、モデル、サイズ、装着可否を保証するものではありません。"
)