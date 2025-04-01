import pandas as pd
import motmetrics as mm
from shapely.geometry import box
import numpy as np

# 修正边界框坐标，防止越界
def correct_bbox_coords(df, img_width=1920, img_height=1080):
    # 计算右下角坐标
    x2 = df['x'] + df['w']
    y2 = df['y'] + df['h']

    # 限制坐标在图像范围内
    df['x'] = df['x'].clip(0, img_width - 1)
    df['y'] = df['y'].clip(0, img_height - 1)
    df['w'] = (x2 - df['x']).clip(lower=1)
    df['h'] = (y2 - df['y']).clip(lower=1)

    return df

# 加载 Ground Truth 和 预测数据
gt_df = pd.read_csv("/home/hzq/projects/Linfer/perf/gt.txt", header=None, names=['frame', 'id', 'x', 'y', 'w', 'h'])
pred_df = pd.read_csv("/home/hzq/projects/Linfer/workspace/track_results.txt", header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis'])

# 修正 Ground Truth 坐标
gt_df = correct_bbox_coords(gt_df)

# 初始化 MOTAccumulator（用来统计计算 MOTA、IDF1 等指标）
acc = mm.MOTAccumulator(auto_id=True)

# 获取所有帧号
frames = sorted(set(gt_df['frame']).union(pred_df['frame']))

# 对每帧进行处理
for frame in frames:
    # 获取当前帧的 GT 和预测框数据
    gt_frame = gt_df[gt_df['frame'] == frame]
    pred_frame = pred_df[pred_df['frame'] == frame]

    # 获取 GT 和预测框的 ID
    gt_ids = list(gt_frame['id'].astype(str))
    pred_ids = list(pred_frame['id'].astype(str))

    # 计算 IOU 距离矩阵
    dists = np.ones((len(gt_ids), len(pred_ids)))  # 默认距离为 1（表示最大距离）

    # 计算每个 GT 和预测框的 IOU 值
    for i, gt_row in gt_frame.iterrows():
        gbox = box(gt_row['x'], gt_row['y'], gt_row['x'] + gt_row['w'], gt_row['y'] + gt_row['h'])
        for j, pred_row in pred_frame.iterrows():
            pbox = box(pred_row['x'], pred_row['y'], pred_row['x'] + pred_row['w'], pred_row['y'] + pred_row['h'])
            iou = gbox.intersection(pbox).area / gbox.union(pbox).area
            dists[gt_frame.index.get_loc(i), pred_frame.index.get_loc(j)] = 1 - iou  # 转换为距离

    # 更新 MOTAccumulator
    acc.update(gt_ids, pred_ids, dists)

# 计算评估指标
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='summary')

# 输出结果
print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
