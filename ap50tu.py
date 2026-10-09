import json
import pandas as pd


# =========================================================
# 只改这三个路径
# =========================================================
BASELINE_LOG = r"baseline.txt"
OURS_LOG = r"ours.txt"

OUTPUT_CSV = r"map_epoch_origin.csv"


def read_log(log_path):
    epochs = []
    ap50_list = []
    map5095_list = []

    with open(log_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # -------------------------------
            # Epoch
            # -------------------------------
            epoch = data.get("epoch", None)

            if epoch is None:
                continue

            # -------------------------------
            # 自动寻找 COCO bbox 评估结果
            # -------------------------------
            coco_result = None

            possible_keys = [
                "test_coco_eval_bbox",
                "coco_eval_bbox",
                "val_coco_eval_bbox",
                "test_bbox"
            ]

            for key in possible_keys:
                if key in data:
                    coco_result = data[key]
                    break

            if coco_result is None:
                continue

            # 至少需要 AP 和 AP50
            if len(coco_result) < 2:
                continue

            # COCOeval:
            # index 0 -> AP@0.5:0.95
            # index 1 -> AP@0.5
            map5095 = float(coco_result[0])
            ap50 = float(coco_result[1])

            epochs.append(epoch)
            ap50_list.append(ap50)
            map5095_list.append(map5095)

    return pd.DataFrame({
        "Epoch": epochs,
        "AP50": ap50_list,
        "mAP5095": map5095_list
    })


# =========================================================
# 读取 Baseline
# =========================================================
baseline = read_log(BASELINE_LOG)

baseline = baseline.rename(columns={
    "AP50": "Baseline_AP50",
    "mAP5095": "Baseline_mAP5095"
})


# =========================================================
# 读取 Ours
# =========================================================
ours = read_log(OURS_LOG)

ours = ours.rename(columns={
    "AP50": "Ours_AP50",
    "mAP5095": "Ours_mAP5095"
})


# =========================================================
# 按 Epoch 合并
# =========================================================
result = pd.merge(
    baseline,
    ours,
    on="Epoch",
    how="outer"
)

result = result.sort_values("Epoch")


# =========================================================
# 保存
# =========================================================
result.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("转换完成：")
print(OUTPUT_CSV)

print()
print(result.head())