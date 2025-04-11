import pandas as pd
import json

# ==== 加载 JSON 的函数 ====
def load_top10_predictions(filepath):
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    top_ids, top_probs = [], []
    for video in raw_data:
        vid_ids, vid_probs = [], []
        for frame in video:
            ids, probs = frame[0], frame[1]
            ids_out, probs_out = [], []
            for cid, prob in zip(ids, probs):
                ids_out.append(cid)
                probs_out.append(prob)
            vid_ids.append(ids_out)
            vid_probs.append(probs_out)
        top_ids.append(vid_ids)
        top_probs.append(vid_probs)
    return top_ids, top_probs

# ==== 加载 class_map.txt ====
def read_class_map(class_map_path):
    class_map = {}
    with open(class_map_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                label, content = line.strip().split(":", 1)
                class_map[int(label.strip())] = content.strip()
    class_map["empty"] = "Unknown"
    return class_map

# ==== 初始化 top5 列 ====
def init_top5_columns():
    return [[] for _ in range(5)]

# ==== 配置路径 ====
csv_path = "data/Sgloss.csv"
json_dom_path = "json/dom_top10_results.json"
json_non_path = "json/non_dom_top10_results.json"
class_map_path = "data/class_map.txt"
output_path = "Summary/Shandshape.csv"

# ==== 加载数据 ====
df = pd.read_csv(csv_path)
df.rename(columns={df.columns[0]: 'key', df.columns[1]: 'start', df.columns[2]: 'end'}, inplace=True)

# ==== 添加 record 编号 ====
record_map = {}
current_record = -1
records = []
for key in df['key']:
    if key not in record_map:
        current_record += 1
        record_map[key] = current_record
    records.append(record_map[key])
df['record'] = records

# ==== 加载 JSON 和 类别映射 ====
top_ids_dom, _ = load_top10_predictions(json_dom_path)
top_ids_non, _ = load_top10_predictions(json_non_path)
class_map = read_class_map(class_map_path)

# ==== 初始化四组列 ====
start_dom_cols = init_top5_columns()
end_dom_cols = init_top5_columns()
start_non_cols = init_top5_columns()
end_non_cols = init_top5_columns()

# ==== 遍历每行数据 ====
for _, row in df.iterrows():
    r = row['record']
    s = int(row['start'])
    e = int(row['end'])

    # dominant
    try:
        start_dom = [class_map.get(cid, class_map["empty"]) for cid in top_ids_dom[r][s][:5]]
    except Exception:
        start_dom = ["Unknown"] * 5
    try:
        end_dom = [class_map.get(cid, class_map["empty"]) for cid in top_ids_dom[r][e][:5]]
    except Exception:
        end_dom = ["Unknown"] * 5

    # non-dominant
    try:
        start_non = [class_map.get(cid, class_map["empty"]) for cid in top_ids_non[r][s][:5]]
    except Exception:
        start_non = ["Unknown"] * 5
    try:
        end_non = [class_map.get(cid, class_map["empty"]) for cid in top_ids_non[r][e][:5]]
    except Exception:
        end_non = ["Unknown"] * 5

    for i in range(5):
        start_dom_cols[i].append(start_dom[i])
        end_dom_cols[i].append(end_dom[i])
        start_non_cols[i].append(start_non[i])
        end_non_cols[i].append(end_non[i])

# ==== 添加列，顺序按组分类 ====
for i in range(5):
    df[f'start_dom_top_{i+1}'] = start_dom_cols[i]
for i in range(5):
    df[f'end_dom_top_{i+1}'] = end_dom_cols[i]
for i in range(5):
    df[f'start_non_top_{i+1}'] = start_non_cols[i]
for i in range(5):
    df[f'end_non_top_{i+1}'] = end_non_cols[i]

# ==== 保存最终结果 ====
df.to_csv(output_path, index=False)
print(f"✅ 输出已保存到：{output_path}")
