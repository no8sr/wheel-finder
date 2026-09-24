from pathlib import Path

import pandas as pd
from PIL import Image

from predict import predict_image


TEST_DIR = Path("dataset/test")
WHEEL_DATA_PATH = Path("wheel_data.csv")

wheel_data = pd.read_csv(WHEEL_DATA_PATH)


def split_values(value):
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split("|")
    ]


def first_value(value):
    values = split_values(value)

    if not values:
        return None

    return values[0]


def basic_attribute_score(candidate, correct_wheel):
    score = 0

    if candidate.get("manufacturer") == correct_wheel.get("manufacturer"):
        score += 15
    else:
        score -= 15

    if str(candidate.get("spoke_count")) == str(
        correct_wheel.get("spoke_count")
    ):
        score += 10
    else:
        score -= 10

    if candidate.get("spoke_type") == correct_wheel.get("spoke_type"):
        score += 10
    else:
        score -= 10

    if candidate.get("construction") == correct_wheel.get("construction"):
        score += 10
    else:
        score -= 10

    return score


def all_attribute_score(candidate, correct_wheel):
    score = basic_attribute_score(
        candidate,
        correct_wheel
    )

    selected_inch = first_value(
        correct_wheel.get("inches")
    )

    selected_holes = first_value(
        correct_wheel.get("holes")
    )

    selected_pcd = first_value(
        correct_wheel.get("pcds")
    )

    selected_color = first_value(
        correct_wheel.get("colors")
    )

    if selected_inch is not None:
        candidate_inches = split_values(
            candidate.get("inches")
        )

        if selected_inch in candidate_inches:
            score += 5
        else:
            score -= 5

    if selected_holes is not None:
        candidate_holes = split_values(
            candidate.get("holes")
        )

        if selected_holes in candidate_holes:
            score += 8
        else:
            score -= 8

    if selected_pcd is not None:
        candidate_pcds = split_values(
            candidate.get("pcds")
        )

        if selected_pcd in candidate_pcds:
            score += 8
        else:
            score -= 6

    if candidate.get("pierce_bolt") == correct_wheel.get("pierce_bolt"):
        score += 5
    else:
        score -= 3

    if selected_color is not None:
        candidate_colors = split_values(
            candidate.get("colors")
        )

        if selected_color in candidate_colors:
            score += 3
        else:
            score -= 1

    return score


def get_wheel_row(class_name):
    rows = wheel_data[
        wheel_data["class_name"] == class_name
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def create_rankings(image):
    cnn_results = predict_image(
        image,
        top_k=6
    )

    image_only_ranking = sorted(
        cnn_results,
        key=lambda item: item.get("probability"),
        reverse=True
    )

    return cnn_results, image_only_ranking


def rank_with_attributes(
    cnn_results,
    correct_wheel,
    score_function
):
    scored_results = []

    for result in cnn_results:
        class_name = result.get("class_name")
        probability = result.get("probability")

        candidate_wheel = get_wheel_row(
            class_name
        )

        if candidate_wheel is None:
            attribute_score = 0
        else:
            attribute_score = score_function(
                candidate_wheel,
                correct_wheel
            )

        final_score = (
            probability * 100
            + attribute_score
        )

        scored_results.append(
            {
                "class_name": class_name,
                "final_score": final_score
            }
        )

    scored_results.sort(
        key=lambda item: item.get("final_score"),
        reverse=True
    )

    return scored_results


def is_top1_correct(ranking, correct_class):
    return (
        ranking[0].get("class_name")
        == correct_class
    )


def is_top3_correct(ranking, correct_class):
    top3_classes = [
        result.get("class_name")
        for result in ranking[:3]
    ]

    return correct_class in top3_classes


total_count = 0

image_top1_count = 0
image_top3_count = 0

basic_top1_count = 0
basic_top3_count = 0

all_top1_count = 0
all_top3_count = 0

class_results = {}
detail_results = []

for class_dir in sorted(TEST_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    correct_class = class_dir.name
    correct_wheel = get_wheel_row(
        correct_class
    )

    if correct_wheel is None:
        print(
            f"CSVに情報がないためスキップ: "
            f"{correct_class}"
        )
        continue

    class_results[correct_class] = {
        "total": 0,
        "image_top1": 0,
        "basic_top1": 0,
        "all_top1": 0
    }

    for image_path in sorted(class_dir.iterdir()):
        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        }:
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        cnn_results, image_ranking = (
            create_rankings(image)
        )

        basic_ranking = rank_with_attributes(
            cnn_results,
            correct_wheel,
            basic_attribute_score
        )

        all_ranking = rank_with_attributes(
            cnn_results,
            correct_wheel,
            all_attribute_score
        )

        image_prediction = image_ranking[0].get(
            "class_name"
        )

        basic_prediction = basic_ranking[0].get(
            "class_name"
        )

        all_prediction = all_ranking[0].get(
            "class_name"
        )

        image_correct = (
            image_prediction == correct_class
        )

        basic_correct = (
            basic_prediction == correct_class
        )

        all_correct = (
            all_prediction == correct_class
        )

        if not image_correct and basic_correct:
            result_type = "基本条件によって改善"

        elif not image_correct and all_correct:
            result_type = "全条件によって改善"

        elif image_correct and not basic_correct:
            result_type = "条件反映によって悪化"

        elif not image_correct and not basic_correct:
            result_type = "条件反映後も誤分類"

        else:
            result_type = "画像のみで正解"

        detail_results.append(
            {
                "image_name": image_path.name,
                "correct_class": correct_class,
                "image_prediction": image_prediction,
                "basic_prediction": basic_prediction,
                "all_prediction": all_prediction,
                "result_type": result_type
            }
        )

        total_count += 1
        class_results[correct_class]["total"] += 1

        if is_top1_correct(
            image_ranking,
            correct_class
        ):
            image_top1_count += 1
            class_results[correct_class][
                "image_top1"
            ] += 1

        if is_top3_correct(
            image_ranking,
            correct_class
        ):
            image_top3_count += 1

        if is_top1_correct(
            basic_ranking,
            correct_class
        ):
            basic_top1_count += 1
            class_results[correct_class][
                "basic_top1"
            ] += 1

        if is_top3_correct(
            basic_ranking,
            correct_class
        ):
            basic_top3_count += 1

        if is_top1_correct(
            all_ranking,
            correct_class
        ):
            all_top1_count += 1
            class_results[correct_class][
                "all_top1"
            ] += 1

        if is_top3_correct(
            all_ranking,
            correct_class
        ):
            all_top3_count += 1


def percentage(correct_count):
    if total_count == 0:
        return 0.0

    return correct_count / total_count * 100


print("\n属性情報の効果比較")
print("=" * 50)
print(f"テスト画像数: {total_count}")

print("\n画像のみ")
print(
    f"Top-1正解率: "
    f"{percentage(image_top1_count):.1f}% "
    f"({image_top1_count}/{total_count})"
)
print(
    f"Top-3正解率: "
    f"{percentage(image_top3_count):.1f}% "
    f"({image_top3_count}/{total_count})"
)

print("\n画像＋基本条件")
print(
    f"Top-1正解率: "
    f"{percentage(basic_top1_count):.1f}% "
    f"({basic_top1_count}/{total_count})"
)
print(
    f"Top-3正解率: "
    f"{percentage(basic_top3_count):.1f}% "
    f"({basic_top3_count}/{total_count})"
)

print("\n画像＋全条件")
print(
    f"Top-1正解率: "
    f"{percentage(all_top1_count):.1f}% "
    f"({all_top1_count}/{total_count})"
)
print(
    f"Top-3正解率: "
    f"{percentage(all_top3_count):.1f}% "
    f"({all_top3_count}/{total_count})"
)

print("\nクラス別Top-1正解数")
print("=" * 50)

for class_name, result in class_results.items():
    print(f"\n{class_name}")
    print(
        f"  画像のみ: "
        f"{result.get('image_top1')}/"
        f"{result.get('total')}"
    )
    print(
        f"  基本条件あり: "
        f"{result.get('basic_top1')}/"
        f"{result.get('total')}"
    )
    print(
        f"  全条件あり: "
        f"{result.get('all_top1')}/"
        f"{result.get('total')}"
    )

print("\n画像ごとの順位変化")
print("=" * 70)

result_types = (
    "基本条件によって改善",
    "全条件によって改善",
    "条件反映によって悪化",
    "条件反映後も誤分類",
)

for result_type in result_types:
    selected_results = [
        result
        for result in detail_results
        if result.get("result_type") == result_type
    ]

    print(f"\n{result_type}: {len(selected_results)}枚")

    for result in selected_results:
        print(
            f"  画像: {result.get('image_name')}"
        )
        print(
            f"    正解: "
            f"{result.get('correct_class')}"
        )
        print(
            f"    画像のみ: "
            f"{result.get('image_prediction')}"
        )
        print(
            f"    基本条件後: "
            f"{result.get('basic_prediction')}"
        )
        print(
            f"    全条件後: "
            f"{result.get('all_prediction')}"
        )

detail_dataframe = pd.DataFrame(
    detail_results
)

detail_dataframe.to_csv(
    "attribute_evaluation_details.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n画像ごとの詳細結果を保存しました: "
    "attribute_evaluation_details.csv"
)   