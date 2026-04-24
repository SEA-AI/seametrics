import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

_HOTA_THRESHOLDS = np.arange(0.05, 0.95 + 1e-9, 0.05)  # 19 values: 0.05 … 0.95


def _iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Vectorised IoU between every (gt, pred) pair. Both arrays are (N, 4) xyxy."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

    gt = gt_boxes[:, None, :]    # (M, 1, 4)
    pr = pred_boxes[None, :, :]  # (1, N, 4)

    inter = (
        np.maximum(0, np.minimum(gt[..., 2], pr[..., 2]) - np.maximum(gt[..., 0], pr[..., 0]))
        * np.maximum(0, np.minimum(gt[..., 3], pr[..., 3]) - np.maximum(gt[..., 1], pr[..., 1]))
    )
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    area_pr = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union = area_gt[:, None] + area_pr[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _hungarian_match(iou_mat: np.ndarray, threshold: float):
    """
    Optimal one-to-one matching via the Hungarian algorithm.
    Returns (matches, unmatched_gt_indices, unmatched_pred_indices).
    matches is a list of (gt_idx, pred_idx, iou).
    Pairs whose IoU is below threshold are discarded.
    """
    M, N = iou_mat.shape
    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))

    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    matched_gt, matched_pr = set(), set()
    matches = []
    for r, c in zip(row_ind, col_ind):
        iou = float(iou_mat[r, c])
        if iou >= threshold:
            matches.append((int(r), int(c), iou))
            matched_gt.add(int(r))
            matched_pr.add(int(c))

    unmatched_gt = [i for i in range(M) if i not in matched_gt]
    unmatched_pr = [j for j in range(N) if j not in matched_pr]
    return matches, unmatched_gt, unmatched_pr


class HOTAMetrics:
    """
    HOTA (Higher Order Tracking Accuracy).

    Reference: Luiten et al., "HOTA: A Higher Order Metric for Evaluating
    Multi-Object Tracking", IJCV 2021.

    Interface mirrors TrackingMetrics so it can be used as a drop-in with
    compute_metrics_by_sequence / hota_results_to_df.

    Input arrays follow the same MOT format used by TrackingMetrics:
        [frame_id, obj_id, x1, y1, x2, y2, confidence, ...]
    """

    def __init__(self, **kwargs) -> None:
        self.accumulators: dict = {}   # sequence_name -> (gt_array, pred_array)
        self.iou_thresholds: np.ndarray = _HOTA_THRESHOLDS
        self.failed_sequences: dict = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, gt: np.ndarray, pred: np.ndarray, sequence_name: str) -> None:
        self.accumulators[sequence_name] = (gt, pred)

    def compute(self, sequence: str = None) -> dict:
        """
        Compute HOTA metrics.

        Parameters
        ----------
        sequence:
            Sequence name to compute metrics for. If None, averages over all
            stored sequences (standard MOT benchmark convention).

        Returns
        -------
        dict with keys: hota, deta, assa, loca  (values in [0, 1])
        """
        if sequence is None:
            per_seq = [
                self._compute_hota(gt, pred)
                for gt, pred in self.accumulators.values()
            ]
            if not per_seq:
                return {}
            keys = per_seq[0].keys()
            return {
                k: float(np.nanmean([r[k] for r in per_seq]))
                for k in keys
            }

        if sequence not in self.accumulators:
            raise KeyError(f"Unknown sequence: {sequence}")
        gt, pred = self.accumulators[sequence]
        return self._compute_hota(gt, pred)

    def log_failed_sequence(self, sequence_name: str, gt, pred) -> None:
        if len(gt) == 0 and len(pred) == 0:
            self.failed_sequences[sequence_name] = "No ground truth and no predictions"
        elif len(gt) == 0:
            self.failed_sequences[sequence_name] = "No ground truth"
        elif len(pred) == 0:
            self.failed_sequences[sequence_name] = "No predictions"
        else:
            self.failed_sequences[sequence_name] = "Missing IDs from GT or Pred"

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_hota(self, gt: np.ndarray, pred: np.ndarray) -> dict:
        nan_result = {k: float("nan") for k in ["hota", "deta", "assa", "loca"]}

        if len(gt) == 0 and len(pred) == 0:
            return nan_result

        # Pre-compute IoU matrices once per frame (reused across all thresholds)
        frames = set()
        if len(gt) > 0:
            frames.update(gt[:, 0].astype(int).tolist())
        if len(pred) > 0:
            frames.update(pred[:, 0].astype(int).tolist())

        frame_cache = []
        for frame in sorted(frames):
            gt_f = gt[gt[:, 0] == frame] if len(gt) > 0 else np.empty((0, 7))
            pr_f = pred[pred[:, 0] == frame] if len(pred) > 0 else np.empty((0, 7))
            gt_ids = gt_f[:, 1].astype(int).tolist()
            pr_ids = pr_f[:, 1].astype(int).tolist()
            gt_boxes = gt_f[:, 2:6] if len(gt_f) > 0 else np.empty((0, 4))
            pr_boxes = pr_f[:, 2:6] if len(pr_f) > 0 else np.empty((0, 4))
            iou_mat = _iou_matrix(gt_boxes, pr_boxes)
            frame_cache.append((gt_ids, pr_ids, iou_mat))

        hota_vals, deta_vals, assa_vals, loca_vals = [], [], [], []

        for alpha in self.iou_thresholds:
            tp_list = []   # (gt_id, pred_id, iou)
            n_fp = 0
            n_fn = 0

            for gt_ids, pr_ids, iou_mat in frame_cache:
                n_gt, n_pr = len(gt_ids), len(pr_ids)
                if n_gt == 0 and n_pr == 0:
                    continue
                if n_gt == 0:
                    n_fp += n_pr
                    continue
                if n_pr == 0:
                    n_fn += n_gt
                    continue

                matches, unmatched_gt, unmatched_pr = _hungarian_match(iou_mat, alpha)
                for gi, pi, iou in matches:
                    tp_list.append((gt_ids[gi], pr_ids[pi], iou))
                n_fp += len(unmatched_pr)
                n_fn += len(unmatched_gt)

            n_tp = len(tp_list)
            total = n_tp + n_fp + n_fn
            deta = n_tp / total if total > 0 else 0.0

            if n_tp == 0:
                assa, loca = 0.0, 0.0
            else:
                pair_counts: dict = defaultdict(int)
                gt_counts: dict = defaultdict(int)
                pr_counts: dict = defaultdict(int)
                for g, p, _ in tp_list:
                    pair_counts[(g, p)] += 1
                    gt_counts[g] += 1
                    pr_counts[p] += 1

                ass_sum = sum(
                    pair_counts[(g, p)] / (gt_counts[g] + pr_counts[p] - pair_counts[(g, p)])
                    for g, p, _ in tp_list
                )
                assa = ass_sum / n_tp
                loca = sum(iou for _, _, iou in tp_list) / n_tp

            hota_vals.append((deta * assa) ** 0.5)
            deta_vals.append(deta)
            assa_vals.append(assa)
            loca_vals.append(loca)

        return {
            "hota": float(np.mean(hota_vals)),
            "deta": float(np.mean(deta_vals)),
            "assa": float(np.mean(assa_vals)),
            "loca": float(np.mean(loca_vals)),
        }
