import torch
from typing import Dict, Tuple, Iterator, Set, List, Optional, cast
from torch import Tensor

from torchmetrics.functional.detection._panoptic_quality_common import (
    _calculate_iou,
    _get_color_areas,
    _Color
)

def _filter_false_negatives(
    target_areas: Dict[_Color, Tensor],
    target_segment_matched: Set[_Color],
    intersection_areas: Dict[Tuple[_Color, _Color], Tensor],
    void_color: Tuple[int, int],
    area: Tuple[float]
) -> Iterator[int]:
    """Filter false negative segments and yield their category IDs.

    False negatives occur when a ground truth segment is not matched with a prediction.
    Areas that are mostly void in the prediction are ignored.

    Args:
        target_areas: Mapping from colors of the ground truth segments to their extents.
        target_segment_matched: Set of ground truth segments that have been matched to a prediction.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Yields:
        Category IDs of segments that account for false negatives.

    """
    lower, upper = area
    false_negative_colors = set(target_areas) - target_segment_matched
    false_negative_colors.discard(void_color)
    for target_color in false_negative_colors:
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        if void_target_area / target_areas[target_color] <= 0.5 and (target_areas[target_color] >= lower) and (target_areas[target_color] < upper):
            yield target_color[0]

def _filter_false_positives(
    pred_areas: Dict[_Color, Tensor],
    pred_segment_matched: Set[_Color],
    intersection_areas: Dict[Tuple[_Color, _Color], Tensor],
    void_color: Tuple[int, int],
    area: Tuple[float]
) -> Iterator[int]:
    """Filter false positive segments and yield their category IDs.

    False positives occur when a predicted segment is not matched with a corresponding target one.
    Areas that are mostly void in the target are ignored.

    Args:
        pred_areas: Mapping from colors of the predicted segments to their extents.
        pred_segment_matched: Set of predicted segments that have been matched to a ground truth.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Yields:
        Category IDs of segments that account for false positives.

    """
    lower, upper = area
    false_positive_colors = set(pred_areas) - pred_segment_matched
    false_positive_colors.discard(void_color)
    for pred_color in false_positive_colors:
        print(pred_areas[pred_color])
        print(area)
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        # we only calculate a prediction as false positive if it is within the current area range
        if pred_void_area / pred_areas[pred_color] <= 0.5 and (pred_areas[pred_color] >= lower) and (pred_areas[pred_color] < upper):
            yield pred_color[0]


def _panoptic_quality_update_sample(
    flatten_preds: Tensor,
    flatten_target: Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    void_color: Tuple[int, int],
    areas: List[Tuple[float]]=None,
    stuffs_modified_metric: Optional[Set[int]] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate stat scores required to compute the metric **for a single sample**.

    Computed scores: iou sum, true positives, false positives, false negatives.

    NOTE: For the modified PQ case, this implementation uses the `true_positives` output tensor to aggregate the actual
        TPs for things classes, but the number of target segments for stuff classes.
        The `iou_sum` output tensor, instead, aggregates the IoU values at different thresholds (i.e., 0.5 for things
        and 0 for stuffs).
        This allows seamlessly using the same `.compute()` method for both PQ variants.

    Args:
        flatten_preds: A flattened prediction tensor referring to a single sample, shape (num_points, 2).
        flatten_target: A flattened target tensor referring to a single sample, shape (num_points, 2).
        cat_id_to_continuous_id: Mapping from original category IDs to continuous IDs
        void_color: an additional, unused color.
        stuffs_modified_metric: Set of stuff category IDs for which the PQ metric is computed using the "modified"
            formula. If not specified, the original formula is used for all categories.

    Returns:
        - IOU Sum
        - True positives
        - False positives
        - False negatives.

    """
    stuffs_modified_metric = stuffs_modified_metric or set()
    device = flatten_preds.device
    num_categories = len(cat_id_to_continuous_id)
    iou_sum = torch.zeros(len(areas), num_categories, dtype=torch.double, device=device)
    true_positives = torch.zeros(len(areas), num_categories, dtype=torch.int, device=device)
    false_positives = torch.zeros(len(areas), num_categories, dtype=torch.int, device=device)
    false_negatives = torch.zeros(len(areas), num_categories, dtype=torch.int, device=device)

    # calculate the area of each prediction, ground truth and pairwise intersection.
    # NOTE: mypy needs `cast()` because the annotation for `_get_color_areas` is too generic.
    pred_areas = cast(Dict[_Color, Tensor], _get_color_areas(flatten_preds))
    target_areas = cast(Dict[_Color, Tensor], _get_color_areas(flatten_target))

    target_areas_split = [
        [target_color for target_color, value in target_areas.items() if (value.item() >= lower and value.item() < upper)]
        for lower, upper in areas
    ]
    print(target_areas_split)

    for i, (target_colors, area) in enumerate(zip(target_areas_split, areas)):
        #for target_areas in target_areas_split:
        # intersection matrix of shape [num_pixels, 2, 2]

        intersection_matrix = torch.transpose(torch.stack((flatten_preds, flatten_target), -1), -1, -2)
        intersection_areas = cast(Dict[Tuple[_Color, _Color], Tensor], _get_color_areas(intersection_matrix))

        # select intersection of things of same category with iou > 0.5
        pred_segment_matched = set()
        target_segment_matched = set()
        for pred_color, target_color in intersection_areas:

            # test only non void, matching category
            if target_color == void_color:
                continue
            if target_color not in target_colors:
                continue
            if pred_color[0] != target_color[0]:
                continue
            iou = _calculate_iou(pred_color, target_color, pred_areas, target_areas, intersection_areas, void_color)
            continuous_id = cat_id_to_continuous_id[target_color[0]]
            if target_color[0] not in stuffs_modified_metric and iou > 0.5:
                pred_segment_matched.add(pred_color)
                target_segment_matched.add(target_color)
                iou_sum[i, continuous_id] += iou
                true_positives[i, continuous_id] += 1
            elif target_color[0] in stuffs_modified_metric and iou > 0:
                iou_sum[i, continuous_id] += iou

        for cat_id in _filter_false_negatives(target_areas, target_segment_matched, intersection_areas, void_color, area=area):
            if cat_id not in stuffs_modified_metric:
                continuous_id = cat_id_to_continuous_id[cat_id]
                false_negatives[i, continuous_id] += 1

        for cat_id in _filter_false_positives(pred_areas, pred_segment_matched, intersection_areas, void_color, area=area):
            if cat_id not in stuffs_modified_metric:
                continuous_id = cat_id_to_continuous_id[cat_id]
                false_positives[i, continuous_id] += 1

        for cat_id, _ in target_areas:
            if cat_id in stuffs_modified_metric:
                continuous_id = cat_id_to_continuous_id[cat_id]
                true_positives[i, continuous_id] += 1
    
    return iou_sum, true_positives, false_positives, false_negatives


def _panoptic_quality_update(
    flatten_preds: Tensor,
    flatten_target: Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    void_color: Tuple[int, int],
    modified_metric_stuffs: Optional[Set[int]] = None,
    areas: List[Tuple[float]] = [(0, 1e10)],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate stat scores required to compute the metric for a full batch.

    Computed scores: iou sum, true positives, false positives, false negatives.

    Args:
        flatten_preds: A flattened prediction tensor, shape (B, num_points, 2).
        flatten_target: A flattened target tensor, shape (B, num_points, 2).
        cat_id_to_continuous_id: Mapping from original category IDs to continuous IDs.
        void_color: an additional, unused color.
        modified_metric_stuffs: Set of stuff category IDs for which the PQ metric is computed using the "modified"
            formula. If not specified, the original formula is used for all categories.

    Returns:
        - IOU Sum
        - True positives
        - False positives
        - False negatives

    """
    device = flatten_preds.device
    num_categories = len(cat_id_to_continuous_id)
    iou_sum = torch.zeros(len(areas), num_categories, dtype=torch.double, device=device)
    true_positives = torch.zeros(len(areas), num_categories, dtype=torch.int, device=device)
    false_positives = torch.zeros(len(areas), num_categories, dtype=torch.int, device=device)
    false_negatives = torch.zeros(len(areas), num_categories, dtype=torch.int, device=device)

    # Loop over each sample independently: segments must not be matched across frames.
    for flatten_preds_single, flatten_target_single in zip(flatten_preds, flatten_target):
        result = _panoptic_quality_update_sample(
            flatten_preds_single,
            flatten_target_single,
            cat_id_to_continuous_id,
            void_color,
            stuffs_modified_metric=modified_metric_stuffs,
            areas=areas
        )
        iou_sum += result[0]
        true_positives += result[1]
        false_positives += result[2]
        false_negatives += result[3]

    return iou_sum, true_positives, false_positives, false_negatives


def _panoptic_quality_compute(
    iou_sum: Tensor,
    true_positives: Tensor,
    false_positives: Tensor,
    false_negatives: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute the final panoptic quality from interim values.

    Args:
        iou_sum: the iou sum from the update step
        true_positives: the TP value from the update step
        false_positives: the FP value from the update step
        false_negatives: the FN value from the update step

    Returns:
        A tuple containing the per-class panoptic, segmentation and recognition quality followed by the averages

    """
    # compute segmentation and recognition quality (per-class)
    sq: Tensor = torch.where(true_positives > 0.0, iou_sum / true_positives, 0.0)
    denominator: Tensor = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    rq: Tensor = torch.where(denominator > 0.0, true_positives / denominator, 0.0)
    # compute per-class panoptic quality
    pq: Tensor = sq * rq
    # compute averages
    pq_avg: Tensor = torch.mean(pq[denominator > 0], dim=-1)
    sq_avg: Tensor = torch.mean(sq[denominator > 0], dim=-1)
    rq_avg: Tensor = torch.mean(rq[denominator > 0], dim=-1)
    return pq, sq, rq, pq_avg, sq_avg, rq_avg