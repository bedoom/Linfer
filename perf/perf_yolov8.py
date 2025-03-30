import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ------------------ 工具函数 ------------------

def normalize_id(image_id, width):
    return image_id.zfill(width)

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa1, ya1, xa2, ya2 = x1, y1, x1+w1, y1+h1
    xb1, yb1, xb2, yb2 = x2, y2, x2+w2, y2+h2

    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def compute_ap(recalls, precisions):
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    ap = sum((mrec[i+1] - mrec[i]) * mpre[i+1] for i in range(len(mrec)-1))
    return ap


# ------------------ 数据读取 ------------------

def detect_id_width(pred_file):
    with open(pred_file) as f:
        return max(len(line.strip().split(',')[0]) for line in f if line.strip())

def load_ground_truth(gt_file, id_width):
    gt = defaultdict(list)
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(',')
            img_id = normalize_id(parts[0], id_width)
            box = list(map(float, parts[2:6]))
            gt[img_id].append(box)
    return gt

def load_predictions(pred_file, id_width, cls_filter=0, conf_thresh=0.001):
    preds = []
    with open(pred_file) as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            cls_id, conf = int(row[1]), float(row[2])
            if cls_id != cls_filter or conf < conf_thresh:
                continue
            img_id = normalize_id(row[0], id_width)
            box = list(map(float, row[3:7]))
            preds.append((img_id, conf, box))
    return sorted(preds, key=lambda x: x[1], reverse=True)


# ------------------ 匹配逻辑 ------------------

def match_predictions(gt_dict, preds, iou_thresh):
    matched = []
    flags = {k: [False]*len(v) for k, v in gt_dict.items()}
    for img_id, conf, box in preds:
        matched_flag = False
        for i, gt_box in enumerate(gt_dict.get(img_id, [])):
            if not flags[img_id][i] and compute_iou(box, gt_box) >= iou_thresh:
                flags[img_id][i] = True
                matched_flag = True
                break
        matched.append((conf, matched_flag))
    total_gt = sum(len(v) for v in gt_dict.values())
    return matched, total_gt


# ------------------ 评估 & 可视化 ------------------

def evaluate_ap(matched, total_gt):
    tp, fp = 0, 0
    recalls, precisions = [], []

    for _, is_tp in matched:
        tp += is_tp
        fp += not is_tp
        recalls.append(tp / total_gt)
        precisions.append(tp / (tp + fp + 1e-6))

    ap = compute_ap(recalls, precisions)
    return tp, fp, total_gt - tp, recalls, precisions, ap

def plot_pr_curve(recalls, precisions, path="pr_curve.png"):
    plt.figure()
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.grid(True)
    plt.savefig(path)
    print(f"[✓] Saved: {path}")

def plot_f1_curve(matched, total_gt, path="f1_curve.png"):
    tp, fp = 0, 0
    confs, f1s = [], []
    for conf, is_tp in matched:
        tp += is_tp
        fp += not is_tp
        fn = total_gt - tp
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        confs.append(conf)
        f1s.append(f1)
    plt.figure()
    plt.plot(confs, f1s, marker='.')
    plt.xlabel("Confidence")
    plt.ylabel("F1 Score")
    plt.title("F1 vs Confidence")
    plt.grid(True)
    plt.savefig(path)
    print(f"[✓] Saved: {path}")


# ------------------ 主入口 ------------------

def evaluate(gt_file="gt.txt", pred_file="result.txt", use_coco=False):
    id_width = detect_id_width(pred_file)
    gt = load_ground_truth(gt_file, id_width)
    preds = load_predictions(pred_file, id_width)

    iou_list = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)] if use_coco else [0.5]
    aps = []

    for iou in iou_list:
        matched, total_gt = match_predictions(gt, preds, iou)
        tp, fp, fn, recalls, precisions, ap = evaluate_ap(matched, total_gt)

        print(f"\nIoU = {iou:.2f} | GT: {total_gt} | TP: {tp} | FP: {fp} | FN: {fn}")
        print(f"Precision: {tp / (tp + fp + 1e-6):.4f} | Recall: {tp / (tp + fn + 1e-6):.4f} | F1: {2*tp/(2*tp + fp + fn + 1e-6):.4f} | AP: {ap:.4f}")

        if iou == 0.5:
            plot_pr_curve(recalls, precisions)
            plot_f1_curve(matched, total_gt)
        aps.append(ap)

    if len(aps) > 1:
        print(f"\nmAP@[0.5:0.95]: {np.mean(aps):.4f}")


if __name__ == "__main__":
    evaluate(use_coco=True)  # True = mAP@[0.5:0.95]，False = mAP@0.5
