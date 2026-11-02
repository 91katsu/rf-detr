"""把 eval.py 的 metrics JSON 轉成 markdown 表格：python metrics_to_md.py metrics.json [more.json ...]"""

import json
import sys
from pathlib import Path

for json_path in map(Path, sys.argv[1:]):
    d = json.loads(json_path.read_text())
    lines = [f"## {json_path.stem}", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in d["overall"].items()]
    lines += ["", "| id | class | GT | AP50:95 | AR | F1 | P | R |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for cid, m in d["per_class"].items():
        if m["num_gt"] == 0:  # 沒有 GT 的類別（AP=-1）不列
            continue
        name = d["class_names"][int(cid)] if int(cid) < len(d["class_names"]) else "(background slot)"
        lines.append(
            f"| {cid} | {name} | {m['num_gt']} | {m['ap']:.4f} | {m['ar']:.4f} "
            f"| {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |"
        )
    out = json_path.with_suffix(".md")
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")
