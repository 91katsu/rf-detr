"""用 RF-DETR 訓練時的同一套指標，對 YOLO 格式的 ground truth 評分。

兩種用法：

1. 現成的預測檔（其他模型的推論結果）
   python eval.py --gt-dir /mnt/10TB_2024/Katsu/data/rf-detr/fundus_yolo_crop/test \
       --pred-dir /mnt/10TB_2024/Katsu/Result/deployment/results_yolo --pred-format yolo
   python eval.py --gt-dir /mnt/10TB_2024/Katsu/data/rf-detr/fundus_yolo_crop/test \
       --pred-dir /mnt/10TB_2024/Katsu/Result/deployment/results --pred-format xyxy \
       --label-mapping /mnt/10TB_2024/Katsu/Result/deployment/label_mapping.txt

2. RF-DETR checkpoint：先 predict() 再評分
   CUDA_VISIBLE_DEVICES=3 python eval.py --gt-dir /mnt/10TB_2024/Katsu/data/rf-detr/fundus_yolo_crop/test \
       --checkpoint /mnt/10TB_2024/Katsu/Result/rf-detr/fundus_baseline/checkpoint_best_ema.pth \
       --save-pred-dir /mnt/10TB_2024/Katsu/Result/rf-detr/fundus_baseline/results_yolo

預測檔格式（每張圖一個 <圖檔名>.txt，每行一個框）：
    yolo : cls cx cy w h score      （0~1 正規化）
    xyxy : cls x1 y1 x2 y2 score    （像素座標，可搭配 --label-mapping 換類別）

mAP/mAR 用 rfdetr.training.coco_map.OnePassCocoMeanAveragePrecision（vernier），F1/P/R 用
rfdetr.evaluation 的 matching + f1_sweep，設定跟 COCOEvalCallback 一樣，所以數字可以直接跟訓練 log 的
val/mAP_50_95、val/F1 比較。差別：訓練驗證的 GT 會經過 dataset transform，這裡直接讀原始標註；
checkpoint 模式走 predict() 的前處理（resize 成正方形，不補邊），跟部署時一樣。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from rfdetr.evaluation.matching import build_matching_data, init_matching_accumulator, merge_matching_data
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
Sample = dict[str, torch.Tensor]


def load_class_names(data_yaml: Path) -> list[str]:
    """Read class names from a YOLO ``data.yaml`` (``names`` as a list or an ``{id: name}`` map)."""
    names = yaml.safe_load(data_yaml.read_text())["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def list_images(images_dir: Path) -> dict[str, Path]:
    """Return ``{stem: path}`` for every image in *images_dir*, sorted by stem."""
    paths = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return {p.stem: p for p in sorted(paths, key=lambda p: p.stem)}


def image_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` from the image header without decoding pixels."""
    with Image.open(path) as img:
        return img.size


def read_rows(path: Path, num_cols: int) -> np.ndarray:
    """Read a whitespace-separated txt as an ``(N, num_cols)`` float array; a missing or empty file gives ``N = 0``."""
    if not path.exists():
        return np.zeros((0, num_cols), dtype=np.float64)
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, num_cols), dtype=np.float64)
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] < num_cols:
        raise ValueError(f"{path}: expected {num_cols} columns, got {arr.shape[1]}")
    return arr[:, :num_cols]


def cxcywh_norm_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized ``cx cy w h`` boxes to absolute ``x1 y1 x2 y2``."""
    cx, cy, w, h = boxes.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) * [width, height, width, height]


def load_gt(label_path: Path, width: int, height: int) -> Sample:
    """Load one YOLO label file (``cls cx cy w h``) as a target dict with absolute xyxy boxes."""
    rows = read_rows(label_path, 5)
    return {
        "boxes": torch.as_tensor(cxcywh_norm_to_xyxy(rows[:, 1:5], width, height), dtype=torch.float32),
        "labels": torch.as_tensor(rows[:, 0], dtype=torch.int64),
    }


def load_label_mapping(path: Path) -> dict[int, int | None]:
    """Read ``src -> dst`` lines; ``dst`` of ``null``/``none`` means the class is dropped."""
    mapping: dict[int, int | None] = {}
    for line in path.read_text().splitlines():
        if "->" not in line:
            continue
        src, dst = (part.strip() for part in line.split("->"))
        mapping[int(src)] = None if dst.lower() in {"null", "none", ""} else int(dst)
    return mapping


def load_pred_file(
    path: Path, fmt: str, width: int, height: int, mapping: dict[int, int | None] | None
) -> Sample:
    """Load one prediction file (``yolo`` or ``xyxy`` format) as a prediction dict with absolute xyxy boxes."""
    rows = read_rows(path, 6)
    labels, boxes, scores = rows[:, 0].astype(np.int64), rows[:, 1:5], rows[:, 5]
    if fmt == "yolo":
        boxes = cxcywh_norm_to_xyxy(boxes, width, height)
    if mapping is not None:
        unknown = set(labels.tolist()) - set(mapping)
        if unknown:
            raise ValueError(f"{path}: classes {sorted(unknown)} are not in the label mapping")
        mapped = [mapping[int(c)] for c in labels]
        keep = np.asarray([m is not None for m in mapped], dtype=bool)
        labels = np.asarray([m for m in mapped if m is not None], dtype=np.int64)
        boxes, scores = boxes[keep], scores[keep]
    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.as_tensor(scores, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


def predict_with_checkpoint(
    checkpoint: Path, images: dict[str, Path], batch_size: int, threshold: float
) -> dict[str, Sample]:
    """Run ``RFDETR.predict`` over *images* and return ``{stem: prediction dict}``.

    ``threshold=0.0`` keeps every detection PostProcess returns (top ``num_select``), as training evaluation does.
    """
    from rfdetr import RFDETR  # 只有 checkpoint 模式需要載入模型
    from rfdetr.utilities.io import _safe_torch_load

    saved_config = _safe_torch_load(checkpoint).get("model_config") or {}
    if saved_config.get("encoder") == "retfound_dinov2_large":
        # 自訂 encoder：from_checkpoint 不認得這個 model_name，改用 RFDETRRETFound 直接載整個 checkpoint
        from my_backbones.retfound import RFDETRRETFound

        model: RFDETR = RFDETRRETFound(pretrain_weights=str(checkpoint), resolution=saved_config.get("resolution", 504))
    else:
        model = RFDETR.from_checkpoint(str(checkpoint))
    stems = list(images)
    preds: dict[str, Sample] = {}
    for start in tqdm(range(0, len(stems), batch_size), desc="predict"):
        batch = stems[start : start + batch_size]
        detections = model.predict([str(images[s]) for s in batch], threshold=threshold, include_source_image=False, antialias=True)
        for stem, det in zip(batch, detections):
            preds[stem] = {
                "boxes": torch.as_tensor(det.xyxy, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.as_tensor(det.confidence, dtype=torch.float32),
                "labels": torch.as_tensor(det.class_id, dtype=torch.int64),
            }
    return preds


def save_yolo_txt(pred: Sample, path: Path, width: int, height: int) -> None:
    """Write a prediction dict as ``cls cx cy w h score`` lines (the ``results_yolo`` format)."""
    x1, y1, x2, y2 = pred["boxes"].numpy().T
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    lines = [
        f"{c} {a:.6f} {b:.6f} {d:.6f} {e:.6f} {s:.6f}"
        for c, a, b, d, e, s in zip(pred["labels"].tolist(), cx, cy, w, h, pred["scores"].tolist())
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


class _PerClassAP50Metric(OnePassCocoMeanAveragePrecision):
    """``OnePassCocoMeanAveragePrecision`` that also returns per-class AP at IoU 0.50 (``map_50_per_class``).

    The parent averages each class's precision over every IoU threshold (AP50:95); this reads the same precision
    array at the IoU = 0.50 slice only, so the numbers come from the identical evaluation.
    """

    def _reduce_per_class(self, evaluation: Any, prefix: str, classes: list[int]) -> dict[str, torch.Tensor]:
        result = super()._reduce_per_class(evaluation, prefix, classes)
        if not self.class_metrics:
            return result
        iou_index = next(i for i, t in enumerate(self.iou_thresholds) if abs(t - 0.5) < 1e-6)
        precision = np.asarray(evaluation["precision"])  # [IoU, recall, class, area, max_dets]
        ap50: list[float] = []
        for class_index in range(len(classes)):
            p = precision[iou_index, :, class_index, 0, -1]
            valid = p[p > -1]
            ap50.append(float(valid.mean()) if valid.size else -1.0)
        result[f"{prefix}map_50_per_class"] = torch.tensor(ap50, dtype=torch.float32)
        return result


def evaluate(preds: list[Sample], targets: list[Sample], max_dets: int, chunk: int = 64) -> dict[str, Any]:
    """Compute mAP/mAR and the F1 sweep exactly as ``COCOEvalCallback`` does for one split."""
    metric = _PerClassAP50Metric(
        iou_type="bbox",
        class_metrics=True,
        max_detection_thresholds=[1, 10, max_dets],
        backend="vernier",
        sync_on_compute=False,
    )
    matching = init_matching_accumulator()
    for start in tqdm(range(0, len(preds), chunk), desc="match"):
        p, t = preds[start : start + chunk], targets[start : start + chunk]
        metric.update(p, t)
        merge_matching_data(matching, build_matching_data(p, t, iou_threshold=0.5, iou_type="bbox"))
    result = metric.compute()

    mar_key = f"mar_{max_dets}"
    classes = result["classes"].tolist()
    per_class: dict[int, dict[str, float]] = {
        cid: {"ap": float(ap), "ap50": float(ap50), "ar": float(ar)}
        for cid, ap, ap50, ar in zip(
            classes,
            result["map_per_class"].tolist(),
            result["map_50_per_class"].tolist(),
            result[f"{mar_key}_per_class"].tolist(),
        )
    }

    # F1 sweep: same as COCOEvalCallback._compute_and_log_f1_metrics
    sorted_ids = sorted(matching)
    f1 = {"F1": 0.0, "Precision": 0.0, "Recall": 0.0, "conf_threshold": 0.0}
    if sorted_ids:
        per_class_list = [matching[cid] for cid in sorted_ids]
        classes_with_gt = [i for i, cid in enumerate(sorted_ids) if matching[cid]["total_gt"] > 0]
        sweep = sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt)
        best = max(sweep, key=lambda x: x["macro_f1"])
        f1 = {
            "F1": float(best["macro_f1"]),
            "Precision": float(best["macro_precision"]),
            "Recall": float(best["macro_recall"]),
            "conf_threshold": float(best["confidence_threshold"]),
        }
        for k, cid in enumerate(sorted_ids):
            entry = per_class.setdefault(cid, {})
            entry["f1"] = float(best["per_class_f1"][k])
            entry["precision"] = float(best["per_class_prec"][k])
            entry["recall"] = float(best["per_class_rec"][k])

    gt_counts = torch.cat([t["labels"] for t in targets]).bincount().tolist() if targets else []
    for cid, entry in per_class.items():
        entry["num_gt"] = gt_counts[cid] if cid < len(gt_counts) else 0

    return {
        "overall": {
            "mAP_50_95": float(result["map"]),
            "mAP_50": float(result["map_50"]),
            "mAP_75": float(result["map_75"]),
            f"mAR_{max_dets}": float(result[mar_key]),
            **f1,
        },
        "per_class": per_class,
    }


def format_report(metrics: dict[str, Any], class_names: list[str], header: list[str]) -> str:
    """Render the metrics as a plain-text report."""
    lines = list(header)
    lines.append("")
    lines += [f"{k:<16}{v:.4f}" for k, v in metrics["overall"].items()]
    lines.append("")
    lines.append(f"{'id':>3} {'class':<40}{'GT':>7}{'AP50:95':>9}{'AP50':>8}{'AR':>8}{'F1':>8}{'P':>8}{'R':>8}")
    for cid in sorted(metrics["per_class"]):
        m = metrics["per_class"][cid]
        # RF-DETR 的分類頭多一個 background slot（id = num_classes），訓練評估也會把它算進來；沒有 GT 所以不影響 mAP/F1
        name = class_names[cid] if cid < len(class_names) else "(background slot)"
        # AP/AR = -1 表示這個類別沒有 GT
        lines.append(
            f"{cid:>3} {name:<40}{m.get('num_gt', 0):>7}{m.get('ap', -1):>9.4f}{m.get('ap50', -1):>8.4f}"
            f"{m.get('ar', -1):>8.4f}"
            f"{m.get('f1', 0):>8.4f}{m.get('precision', 0):>8.4f}{m.get('recall', 0):>8.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-dir", type=Path, required=True, help="split 資料夾，底下有 images/ 和 labels/")
    parser.add_argument("--images-dir", type=Path, help="預設 <gt-dir>/images")
    parser.add_argument("--labels-dir", type=Path, help="預設 <gt-dir>/labels")
    parser.add_argument("--data-yaml", type=Path, help="預設 <gt-dir>/../data.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred-dir", type=Path, help="預測 txt 資料夾")
    source.add_argument("--checkpoint", type=Path, help="RF-DETR checkpoint，會先跑 predict()")
    parser.add_argument("--pred-format", choices=["yolo", "xyxy"], default="yolo")
    parser.add_argument("--label-mapping", type=Path, help="預測類別換算表（src -> dst / null）")
    parser.add_argument("--max-dets", type=int, default=500, help="同 TrainConfig.eval_max_dets")
    parser.add_argument("--threshold", type=float, default=0.0, help="checkpoint 模式的 predict() 分數門檻")
    parser.add_argument("--batch-size", type=int, default=8, help="checkpoint 模式的 predict() batch")
    parser.add_argument("--save-pred-dir", type=Path, help="checkpoint 模式：把預測存成 results_yolo 格式")
    parser.add_argument("--output", type=Path, help="metrics JSON 路徑（同名 .txt 會存文字報告）；不給就只印在 terminal")
    return parser.parse_args()


def main() -> None:
    """Load GT and predictions, evaluate, print and save the report."""
    args = parse_args()
    images_dir = args.images_dir or args.gt_dir / "images"
    labels_dir = args.labels_dir or args.gt_dir / "labels"
    data_yaml = args.data_yaml or args.gt_dir.parent / "data.yaml"
    class_names = load_class_names(data_yaml)

    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")
    sizes = {stem: image_size(path) for stem, path in tqdm(images.items(), desc="read sizes")}
    targets = {stem: load_gt(labels_dir / f"{stem}.txt", *sizes[stem]) for stem in images}

    header = [f"gt     : {args.gt_dir}  images={len(images)}  gt boxes={sum(len(t['labels']) for t in targets.values())}"]
    warnings: list[str] = []
    if args.checkpoint is not None:
        preds = predict_with_checkpoint(args.checkpoint, images, args.batch_size, args.threshold)
        header.append(f"source : {args.checkpoint}  (predict threshold={args.threshold})")
        if args.save_pred_dir is not None:
            args.save_pred_dir.mkdir(parents=True, exist_ok=True)
            for stem, pred in preds.items():
                save_yolo_txt(pred, args.save_pred_dir / f"{stem}.txt", *sizes[stem])
            print(f"Saved predictions to {args.save_pred_dir}")
    else:
        mapping = load_label_mapping(args.label_mapping) if args.label_mapping else None
        pred_stems = {p.stem for p in args.pred_dir.glob("*.txt")}
        missing = sorted(set(images) - pred_stems)
        orphans = sorted(pred_stems - set(images))
        if missing:
            warnings.append(f"{len(missing)} images have no prediction file (counted as no detections), e.g. {missing[:3]}")
        if orphans:
            warnings.append(f"{len(orphans)} prediction files have no matching image (ignored), e.g. {orphans[:3]}")
        preds = {
            stem: load_pred_file(args.pred_dir / f"{stem}.txt", args.pred_format, *sizes[stem], mapping)
            for stem in tqdm(images, desc="read preds")
        }
        header.append(f"source : {args.pred_dir}  format={args.pred_format}  mapping={args.label_mapping}")
    header.append(f"dets   : {sum(len(p['labels']) for p in preds.values())}  max_dets={args.max_dets}")
    header += [f"WARNING: {w}" for w in warnings]

    stems = list(images)
    metrics = evaluate([preds[s] for s in stems], [targets[s] for s in stems], args.max_dets)
    report = format_report(metrics, class_names, header)
    print(report)

    output = args.output
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"header": header, **metrics, "class_names": class_names}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    output.with_suffix(".txt").write_text(report + "\n")
    print(f"\nSaved {output} and {output.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
