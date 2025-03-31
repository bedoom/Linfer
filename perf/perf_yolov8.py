import torch
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import csv
from collections import defaultdict
import logging

logging.getLogger("torchmetrics").setLevel(logging.ERROR)

# ------------------ 工具函数 ------------------

def normalize_id(image_id, width):
    """标准化图像ID为固定长度"""
    return image_id.zfill(width)

def convert_to_xyxy(box, img_width, img_height):
    """将 [x, y, w, h] 转换为 [x1, y1, x2, y2]，并确保框在图像内"""
    x, y, w, h = box
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # 确保框在图像内
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_width), min(y2, img_height)
    
    return [x1, y1, x2, y2]

def load_data(file, id_width, img_width, img_height, is_gt=True):
    """加载ground truth或预测数据，并转换框的坐标"""
    data = defaultdict(list) if is_gt else []
    with open(file) as f:
        for row in csv.reader(f):
            if is_gt:
                img_id:str = normalize_id(row[0], id_width)
                box = convert_to_xyxy(list(map(float, row[2:6])), img_width, img_height)
                data[img_id].append(box) # type: ignore
            else:
                cls_id, conf = int(row[1]), float(row[2])
                if conf < 0.001: continue  # 忽略低置信度框
                img_id = normalize_id(row[0], id_width)
                box = convert_to_xyxy(list(map(float, row[3:7])), img_width, img_height)
                data.append((img_id, conf, box)) # type: ignore
    return data

# ------------------ 使用torchmetrics进行评估 ------------------

def evaluate(gt_file="./perf/gt.txt", pred_file="./workspace/yolov8n_result.txt", img_width=1920, img_height=1080):
    """主函数，进行评估并计算多种性能指标"""
    id_width = max(len(line.split(',')[0]) for line in open(pred_file))
    
    # 加载数据
    gt_data = load_data(gt_file, id_width, img_width, img_height, is_gt=True)
    pred_data = load_data(pred_file, id_width, img_width, img_height, is_gt=False)

    # 准备torchmetrics需要的格式
    # all_preds = [{"boxes": torch.tensor([box]), "scores": torch.tensor([conf]), "labels": torch.tensor([0])} 
    #              for img_id, conf, box in pred_data]

    # all_gt = {img_id: [{"boxes": torch.tensor(boxes), "labels": torch.tensor([0] * len(boxes))}]
    #           for img_id, boxes in gt_data.items()} # type: ignore

    pred_dict = defaultdict(list)
    for img_id, conf, box in pred_data:
        pred_dict[img_id].append((conf, box))

    # 所有图片ID（合并预测和GT中的ID）
    img_ids = sorted(set(pred_dict.keys()) | set(gt_data.keys()))

    # 构造 torchmetrics 所需格式
    all_preds = []
    all_gt = []

    for img_id in img_ids:
        preds = pred_dict.get(img_id, [])
        gts = gt_data.get(img_id, [])

        pred_boxes = [box for conf, box in preds]
        pred_scores = [conf for conf, box in preds]

        # 每张图像的预测
        all_preds.append({
            "boxes": torch.tensor(pred_boxes, dtype=torch.float32) if pred_boxes else torch.empty((0, 4)),
            "scores": torch.tensor(pred_scores, dtype=torch.float32) if pred_scores else torch.empty((0,)),
            "labels": torch.zeros(len(pred_boxes), dtype=torch.int64)  # 全部设为类0
        })

        # 每张图像的ground truth
        all_gt.append({
            "boxes": torch.tensor(gts, dtype=torch.float32) if gts else torch.empty((0, 4)),
            "labels": torch.zeros(len(gts), dtype=torch.int64)  # 全部设为类0
        })

    # 使用torchmetrics计算mAP
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(all_preds, all_gt) # type: ignore
    result = metric.compute()

    # 输出各项评估指标
    print(f"mAP@[0.5:0.95]: {result['map']:.4f}")
    print(f"mAP@0.5: {result['map_50']:.4f}")
    print(f"mAP@0.75: {result['map_75']:.4f}")
    # for i, ap in enumerate(result["ap_class"]):
    #     print(f"AP for class {i}: {ap:.4f}")
    # print(f"Precision: {result['precision']:.4f}")
    # print(f"Recall: {result['recall']:.4f}")
    # print(f"F1 Score: {result['f1']:.4f}")

    # 绘制并保存PR曲线
    # plot_pr_curve(result['recall'], result['precision'])

def plot_pr_curve(recall, precision, path="pr_curve.png"):
    """绘制并保存PR曲线"""
    plt.figure()
    plt.plot(recall, precision, marker='o')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.grid(True)
    plt.savefig(path)
    print(f"[✓] Saved: {path}")

if __name__ == "__main__":
    evaluate()
