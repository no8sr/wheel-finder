import pandas as pd
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

@st.cache_data
def load_wheel_data():
    return pd.read_csv("wheel_data.csv")


wheel_data = load_wheel_data()

def split_values(value):
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split("|")
    ]


def condition_matches(
    csv_value,
    selected_value
):
    available_values = split_values(csv_value)
    selected_text = str(selected_value)

    return selected_text in available_values

def calculate_attribute_score(
    wheel,
    selected_manufacturer,
    selected_spoke_count,
    selected_spoke_type,
    selected_construction,
    selected_inch,
    selected_holes,
    selected_pcd,
    selected_pierce_bolt
):
    score = 0
    matched_conditions = []
    mismatched_conditions = []

    # メーカー
    if selected_manufacturer != "指定なし":
        if wheel["manufacturer"] == selected_manufacturer:
            score += 15
            matched_conditions.append("メーカー")
        else:
            score -= 15
            mismatched_conditions.append("メーカー")

    # スポーク数
    if selected_spoke_count != "指定なし":
        if str(wheel["spoke_count"]) == str(selected_spoke_count):
            score += 10
            matched_conditions.append("スポーク数")
        else:
            score -= 10
            mismatched_conditions.append("スポーク数")

    # スポーク構造
    if selected_spoke_type != "指定なし":
        if wheel["spoke_type"] == selected_spoke_type:
            score += 10
            matched_conditions.append("スポーク構造")
        else:
            score -= 10
            mismatched_conditions.append("スポーク構造")

    # ピース構造
    if selected_construction != "指定なし":
        if wheel["construction"] == selected_construction:
            score += 10
            matched_conditions.append("ピース構造")
        else:
            score -= 10
            mismatched_conditions.append("ピース構造")

        # インチ
    if selected_inch != "指定なし":
        if condition_matches(
            wheel["inches"],
            selected_inch
        ):
            score += 5
            matched_conditions.append("インチ")
        else:
            score -= 5
            mismatched_conditions.append("インチ")

    # 穴数
    if selected_holes != "指定なし":
        if condition_matches(
            wheel["holes"],
            selected_holes
        ):
            score += 8
            matched_conditions.append("穴数")
        else:
            score -= 8
            mismatched_conditions.append("穴数")

    # PCD
    if selected_pcd != "指定なし":
        if condition_matches(
            wheel["pcds"],
            selected_pcd
        ):
            score += 8
            matched_conditions.append("PCD")
        else:
            score -= 6
            mismatched_conditions.append("PCD")

    # ピアスボルト
    if selected_pierce_bolt != "指定なし":
        if wheel["pierce_bolt"] == selected_pierce_bolt:
            score += 5
            matched_conditions.append("ピアスボルト")
        else:
            score -= 3
            mismatched_conditions.append("ピアスボルト")

    return (
        score,
        matched_conditions,
        mismatched_conditions
    )

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

st.caption(
    "入力条件は候補順位の調整に使用されます。"
    "不確かな項目は「指定なし」を選択してください。"
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
                "RAYS",
                "WORK",
                "BBS"
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
            "ピース構造",
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

        # 登録されている6モデルすべての予測値を取得
        cnn_results = predict_image(
            image,
            top_k=6
        )

        ranked_results = []

        for result in cnn_results:
            class_name = result["class_name"]
            probability = result["probability"]

            wheel_rows = wheel_data[
                wheel_data["class_name"] == class_name
            ]

            if wheel_rows.empty:
                attribute_score = 0
                matched_conditions = []
                mismatched_conditions = []

            else:
                wheel = wheel_rows.iloc[0]

                score_result = calculate_attribute_score(
                    wheel=wheel,
                    selected_manufacturer=manufacturer,
                    selected_spoke_count=spoke_count,
                    selected_spoke_type=spoke_type,
                    selected_construction=construction,
                    selected_inch=inch,
                    selected_holes=holes,
                    selected_pcd=pcd,
                    selected_pierce_bolt=pierce_bolt
                )

                attribute_score = score_result[0]
                matched_conditions = score_result[1]
                mismatched_conditions = score_result[2]

            image_score = probability * 100
            final_score = image_score + attribute_score

            ranked_results.append(
                {
                    "class_name": class_name,
                    "probability": probability,
                    "image_score": image_score,
                    "attribute_score": attribute_score,
                    "final_score": final_score,
                    "matched_conditions": matched_conditions,
                    "mismatched_conditions": mismatched_conditions
                }
            )

        image_only_results = sorted(
            ranked_results,
            key=lambda item: item["image_score"],
            reverse=True
        )

        # 最終スコアが高い順に並べ替える
        ranked_results.sort(
            key=lambda item: item["final_score"],
            reverse=True
        )

        image_only_first = image_only_results[0]
        final_first = ranked_results[0]

        image_only_name = DISPLAY_NAMES.get(
            image_only_first.get("class_name"),
            image_only_first.get("class_name")
        )

        final_name = DISPLAY_NAMES.get(
            final_first.get("class_name"),
            final_first.get("class_name")
        )

        comparison_col1, comparison_col2 = st.columns(2)

        with comparison_col1:
            st.metric(
                "画像のみの第1候補",
                image_only_name
            )

        with comparison_col2:
            st.metric(
                "条件反映後の第1候補",
                final_name
            )

        image_only_class = image_only_first.get("class_name")
        final_class = final_first.get("class_name")

        if image_only_class != final_class:
            st.info(
                "入力条件を反映したことで、"
                "第1候補が変更されました。"
            )
        else:
            st.caption(
                "入力条件を反映しても、"
                "第1候補は変わりませんでした。"
            )

        # 上位3件だけを表示
        top_results = ranked_results[:3]

        result_columns = st.columns(3)

        for rank, result in enumerate(top_results):
            class_name = result["class_name"]
            display_name = DISPLAY_NAMES.get(
                class_name,
                class_name
            )

            probability = result.get("probability")
            image_score = result.get("image_score")
            attribute_score = result.get("attribute_score")
            final_score = result.get("final_score")
            matched_conditions = result.get(
                "matched_conditions",
                []
            )
            mismatched_conditions = result.get(
                "mismatched_conditions",
                []
            )

            result_column = result_columns[rank]

            with result_column:
                st.write(f"### 第{rank + 1}候補")
                st.write(f"**{display_name}**")

                st.progress(float(probability))

                st.write(
                    f"画像の予測確率："
                    f"{probability * 100:.1f}%"
                )

                st.write(
                    f"入力条件による補正："
                    f"{attribute_score:+d}点"
                )

                st.write(
                    f"最終スコア："
                    f"{final_score:.1f}点"
                )

                if matched_conditions:
                    matched_text = "、".join(
                        matched_conditions
                    )

                    st.success(
                        f"一致した条件：{matched_text}"
                    )

                if mismatched_conditions:
                    mismatched_text = "、".join(
                        mismatched_conditions
                    )

                    st.warning(
                        f"一致しない条件：{mismatched_text}"
                    )

                if (
                    not matched_conditions
                    and not mismatched_conditions
                ):
                    st.caption(
                        "順位調整に使用された入力条件はありません。"
                    )


st.divider()

st.caption(
    "検索結果は画像認識と入力条件に基づく候補です。"
    "製品のメーカー、モデル、サイズ、装着可否を保証するものではありません。"
)