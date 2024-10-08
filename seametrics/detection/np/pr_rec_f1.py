# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: This metric is based on torchmetrics.detection.mean_ap and
# then modified to support the evaluation of precision, recall, f1 and support
# for object detection. It can also be used to evaluate the mean average precision
# but some modifications are needed. Additionally, numpy is used instead of torch

import contextlib
import io
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Literal
import numpy as np
from seametrics.detection.np.utils import (
    _fix_empty_arrays,
    _input_validator,
    box_convert,
)

try:
    import pycocotools.mask as mask_utils
    from pycocotools.coco import COCO

    # from pycocotools.cocoeval import COCOeval
    from ..cocoeval import COCOeval  # use our own version of COCOeval
except ImportError:
    raise ModuleNotFoundError(
        "`MAP` metric requires that `pycocotools` installed."
        " Please install with `pip install pycocotools`"
    )


class PrecisionRecallF1Support:
    r"""Compute the Precision, Recall, F1 and Support scores for object detection.

    - Precision = :math:`\frac{TP}{TP + FP}`
    - Recall = :math:`\frac{TP}{TP + FN}`
    - F1 = :math:`\frac{2 * Precision * Recall}{Precision + Recall}`
    - Support = :math:`TP + FN`

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:
        - boxes: (:class:`~np.ndarray`) of shape ``(num_boxes, 4)`` containing ``num_boxes``
        detection boxes of the format specified in the constructor. By default, this method expects
        ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - scores: :class:`~np.ndarray` of shape ``(num_boxes)`` containing detection scores
        for the boxes.
        - labels: :class:`~np.ndarray` of shape ``(num_boxes)`` containing 0-indexed detection
        classes for the boxes.
        - masks: :class:`~torch.bool` of shape ``(num_boxes, image_height, image_width)`` containing
        boolean masks. Only required when `iou_type="segm"`.

    - ``target`` (:class:`~List`) A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:
        - boxes: :class:`~np.ndarray` of shape ``(num_boxes, 4)`` containing ``num_boxes``
        ground truth boxes of the format specified in the constructor. By default, this method
        expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - labels: :class:`~np.ndarray` of shape ``(num_boxes)`` containing 0-indexed ground
        truth classes for the boxes.
        - masks: :class:`~torch.bool` of shape ``(num_boxes, image_height, image_width)``
        containing boolean masks. Only required when `iou_type="segm"`.
        - iscrowd: :class:`~np.ndarray` of shape ``(num_boxes)`` containing 0/1 values
        indicating whether the bounding box/masks indicate a crowd of objects. Value is optional,
        and if not provided it will automatically be set to 0.
        - area: :class:`~np.ndarray` of shape ``(num_boxes)`` containing the area of the
        object. Value if optional, and if not provided will be automatically calculated based
        on the bounding box/masks provided. Only affects when 'area_ranges' is provided.

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``results``: A dictionary containing the following key-values:

        - ``params``: COCOeval parameters object
        - ``eval``: output of COCOeval.accumuate()
        - ``metrics``: A dictionary containing the following key-values for each area range:
            - ``area_range``: str containing the area range
            - ``iouThr``: str containing the IoU threshold
            - ``maxDets``: int containing the maximum number of detections
            - ``tp``: int containing the number of true positives
            - ``fp``: int containing the number of false positives
            - ``fn``: int containing the number of false negatives
            - ``precision``: float containing the precision
            - ``recall``: float containing the recall
            - ``f1``: float containing the f1 score
            - ``support``: int containing the support (tp + fn)

    .. note::
        This metric utilizes the official `pycocotools` implementation as its backend. This means that the metric
        requires you to have `pycocotools` installed. In addition we require `torchvision` version 0.8.0 or newer.
        Please install with ``pip install torchmetrics[detection]``.

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[xyxy, xywh, cxcywh]``.
        iou_type:
            Type of input (either masks or bounding-boxes) used for computing IOU.
            Supported IOU types are ``["bbox", "segm"]``. If using ``"segm"``, masks should be provided in input.
        iou_thresholds:
            IoU thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0.5,...,0.95]``
            with step ``0.05``. Else provide a list of floats.
        rec_thresholds:
            Recall thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0,...,1]``
            with step ``0.01``. Else provide a list of floats.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use thresholds ``[100]``.
            Else, please provide a list of ints.
        area_ranges:
            Area ranges for evaluation. If set to ``None`` it corresponds to the ranges ``[[0^2, 1e5^2]]``.
            Else, please provide a list of lists of length 2.
        area_ranges_labels:
            Labels for the area ranges. If set to ``None`` it corresponds to the labels ``["all"]``.
            Else, please provide a list of strings of the same length as ``area_ranges``.
        class_agnostic:
            If ``True`` will compute metrics globally. If ``False`` will compute metrics per class.
            Default: ``True`` (per class metrics are not supported yet)
        debug:
            If ``True`` will print the COCOEval summary to stdout.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``box_format`` is not one of ``"xyxy"``, ``"xywh"`` or ``"cxcywh"``
        ValueError:
            If ``iou_type`` is not one of ``"bbox"`` or ``"segm"``
        ValueError:
            If ``iou_thresholds`` is not None or a list of floats
        ValueError:
            If ``rec_thresholds`` is not None or a list of floats
        ValueError:
            If ``max_detection_thresholds`` is not None or a list of ints
        ValueError:
            If ``area_ranges`` is not None or a list of lists of length 2
        ValueError:
            If ``area_ranges_labels`` is not None or a list of strings

    Example:
        >>> import numpy as np
        >>> from metrics.detection import MeanAveragePrecision
        >>> preds = [
        ...   dict(
        ...     boxes=np.array([[258.0, 41.0, 606.0, 285.0]]),
        ...     scores=np.array([0.536]),
        ...     labels=np.array([0]),
        ...   )
        ... ]
        >>> target = [
        ...   dict(
        ...     boxes=np.array([[214.0, 41.0, 562.0, 285.0]]),
        ...     labels=np.array([0]),
        ...   )
        ... ]
        >>> metric = PrecisionRecallF1Support()
        >>> metric.update(preds, target)
        >>> print(metric.compute())
        {'params': <metrics.detection.cocoeval.Params at 0x16dc99150>,
         'eval': ... output of COCOeval.accumuate(),
         'metrics': {'all': {'range': [0, 10000000000.0],
         'iouThr': '0.50',
         'maxDets': 100,
         'tp': 1,
         'fp': 0,
         'fn': 0,
         'precision': 1.0,
         'recall': 1.0,
         'f1': 1.0,
         'support': 1}}}
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    detections: List[np.ndarray]
    detection_scores: List[np.ndarray]
    detection_labels: List[np.ndarray]
    groundtruths: List[np.ndarray]
    groundtruth_labels: List[np.ndarray]
    groundtruth_crowds: List[np.ndarray]
    groundtruth_area: List[np.ndarray]

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_type: Literal["bbox", "segm"] = "bbox",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        area_ranges: Optional[List[List[int]]] = None,
        area_ranges_labels: Optional[List[str]] = None,
        class_agnostic: bool = True,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}"
            )
        self.box_format = box_format

        allowed_iou_types = ("segm", "bbox")
        if iou_type not in allowed_iou_types:
            raise ValueError(
                f"Expected argument `iou_type` to be one of {allowed_iou_types} but got {iou_type}"
            )
        self.iou_type = iou_type

        if iou_thresholds is not None and not isinstance(iou_thresholds, list):
            raise ValueError(
                f"Expected argument `iou_thresholds` to either be `None` or a list of floats but got {iou_thresholds}"
            )
        self.iou_thresholds = (
            iou_thresholds
            or [0.5, 0.75]
        )

        if rec_thresholds is not None and not isinstance(rec_thresholds, list):
            raise ValueError(
                f"Expected argument `rec_thresholds` to either be `None` or a list of floats but got {rec_thresholds}"
            )
        self.rec_thresholds = (
            rec_thresholds or np.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()
        )

        if max_detection_thresholds is not None and not isinstance(
            max_detection_thresholds, list
        ):
            raise ValueError(
                f"Expected argument `max_detection_thresholds` to either be `None` or a list of ints"
                f" but got {max_detection_thresholds}"
            )
        max_det_thr = np.sort(
            np.array(max_detection_thresholds or [100], dtype=np.uint)
        )
        self.max_detection_thresholds = max_det_thr.tolist()

        # check area ranges
        if area_ranges is not None:
            if not isinstance(area_ranges, list):
                raise ValueError(
                    f"Expected argument `area_ranges` to either be `None` or a list of lists but got {area_ranges}"
                )
            for area_range in area_ranges:
                if not isinstance(area_range, list) or len(area_range) != 2:
                    raise ValueError(
                        f"Expected argument `area_ranges` to be a list of lists of length 2 but got {area_ranges}"
                    )
        self.area_ranges = area_ranges if area_ranges is not None else [[0**2, 1e5**2]]

        if area_ranges_labels is not None:
            if area_ranges is None:
                raise ValueError(
                    "Expected argument `area_ranges_labels` to be `None` if `area_ranges` is not provided"
                )
            if not isinstance(area_ranges_labels, list):
                raise ValueError(
                    f"Expected argument `area_ranges_labels` to either be `None` or a list of strings"
                    f" but got {area_ranges_labels}"
                )
            if len(area_ranges_labels) != len(area_ranges):
                raise ValueError(
                    f"Expected argument `area_ranges_labels` to be a list of length {len(area_ranges)}"
                    f" but got {area_ranges_labels}"
                )
        self.area_ranges_labels = (
            area_ranges_labels if area_ranges_labels is not None else ["all"]
        )

        # if not isinstance(class_metrics, bool):
        #     raise ValueError(
        #         "Expected argument `class_metrics` to be a boolean")
        # self.class_metrics = class_metrics

        if not isinstance(class_agnostic, bool):
            raise ValueError("Expected argument `class_agnostic` to be a boolean")
        self.class_agnostic = class_agnostic

        if not isinstance(debug, bool):
            raise ValueError("Expected argument `debug` to be a boolean")
        self.debug = debug

        self.detections = []
        self.detection_scores = []
        self.detection_labels = []
        self.groundtruths = []
        self.groundtruth_labels = []
        self.groundtruth_crowds = []
        self.groundtruth_area = []

        # self.add_state("detections", default=[], dist_reduce_fx=None)
        # self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        # self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        # self.add_state("groundtruths", default=[], dist_reduce_fx=None)
        # self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        # self.add_state("groundtruth_crowds", default=[], dist_reduce_fx=None)
        # self.add_state("groundtruth_area", default=[], dist_reduce_fx=None)

    def update(
        self, preds: List[Dict[str, np.ndarray]], target: List[Dict[str, np.ndarray]]
    ) -> None:
        """Update metric state.

        Raises:
            ValueError:
                If ``preds`` is not of type (:class:`~List[Dict[str, np.ndarray]]`)
            ValueError:
                If ``target`` is not of type ``List[Dict[str, np.ndarray]]``
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        _input_validator(preds, target, iou_type=self.iou_type)

        for item in preds:
            detections = self._get_safe_item_values(item)

            self.detections.append(detections)
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            groundtruths = self._get_safe_item_values(item)
            self.groundtruths.append(groundtruths)
            self.groundtruth_labels.append(item["labels"])
            self.groundtruth_crowds.append(
                item.get("iscrowd", np.zeros_like(item["labels"]))
            )
            self.groundtruth_area.append(
                item.get("area", np.zeros_like(item["labels"]))
            )

    def compute(self, meas_type='pr_rec_f1') -> dict:
        """Computes the metrics for meas_type: ('pr_rec_f1', 'ap', or 'ar')."""
        coco_target, coco_preds = COCO(), COCO()

        coco_target.dataset = self._get_coco_format(
            self.groundtruths,
            self.groundtruth_labels,
            crowds=self.groundtruth_crowds,
            area=self.groundtruth_area,
        )
        coco_preds.dataset = self._get_coco_format(
            self.detections, self.detection_labels, scores=self.detection_scores
        )

        with contextlib.redirect_stdout(io.StringIO()) as f:
            coco_target.createIndex()
            coco_preds.createIndex()

            coco_eval = COCOeval(coco_target, coco_preds, iouType=self.iou_type)
            coco_eval.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
            coco_eval.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
            coco_eval.params.maxDets = self.max_detection_thresholds
            coco_eval.params.areaRng = self.area_ranges
            coco_eval.params.areaRngLbl = self.area_ranges_labels
            coco_eval.params.useCats = 0 if self.class_agnostic else 1

            coco_eval.evaluate()
            coco_eval.accumulate()

        if self.debug:
            print(f.getvalue())

        metrics = coco_eval.summarize(meas_type)
        return metrics
    
    @staticmethod
    def coco_to_np(
        coco_preds: str,
        coco_target: str,
        iou_type: Literal["bbox", "segm"] = "bbox",
    ) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
        """Utility function for converting .json coco format files to the input format of this metric.

        The function accepts a file for the predictions and a file for the target in coco format and converts them to
        a list of dictionaries containing the boxes, labels and scores in the input format of this metric.

        Args:
            coco_preds: Path to the json file containing the predictions in coco format
            coco_target: Path to the json file containing the targets in coco format
            iou_type: Type of input, either `bbox` for bounding boxes or `segm` for segmentation masks

        Returns:
            preds: List of dictionaries containing the predictions in the input format of this metric
            target: List of dictionaries containing the targets in the input format of this metric

        Example:
            >>> # File formats are defined at https://cocodataset.org/#format-data
            >>> # Example files can be found at
            >>> # https://github.com/cocodataset/cocoapi/tree/master/results
            >>> from torchmetrics.detection import MeanAveragePrecision
            >>> preds, target = MeanAveragePrecision.coco_to_tm(
            ...   "instances_val2014_fakebbox100_results.json.json",
            ...   "val2014_fake_eval_res.txt.json"
            ...   iou_type="bbox"
            ... )  # doctest: +SKIP

        """
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(coco_target)
            dt = gt.loadRes(coco_preds)

        gt_dataset = gt.dataset["annotations"]
        dt_dataset = dt.dataset["annotations"]

        target = {}
        for t in gt_dataset:
            if t["image_id"] not in target:
                target[t["image_id"]] = {
                    "boxes" if iou_type == "bbox" else "masks": [],
                    "labels": [],
                    "iscrowd": [],
                    "area": [],
                }
            if iou_type == "bbox":
                target[t["image_id"]]["boxes"].append(t["bbox"])
            else:
                target[t["image_id"]]["masks"].append(gt.annToMask(t))
            target[t["image_id"]]["labels"].append(t["category_id"])
            target[t["image_id"]]["iscrowd"].append(t["iscrowd"])
            target[t["image_id"]]["area"].append(t["area"])

        preds = {}
        for p in dt_dataset:
            if p["image_id"] not in preds:
                preds[p["image_id"]] = {
                    "boxes" if iou_type == "bbox" else "masks": [],
                    "scores": [],
                    "labels": [],
                }
            if iou_type == "bbox":
                preds[p["image_id"]]["boxes"].append(p["bbox"])
            else:
                preds[p["image_id"]]["masks"].append(gt.annToMask(p))
            preds[p["image_id"]]["scores"].append(p["score"])
            preds[p["image_id"]]["labels"].append(p["category_id"])
        for k in target:  # add empty predictions for images without predictions
            if k not in preds:
                preds[k] = {
                    "boxes" if iou_type == "bbox" else "masks": [],
                    "scores": [],
                    "labels": [],
                }

        batched_preds, batched_target = [], []
        for key in target:
            name = "boxes" if iou_type == "bbox" else "masks"
            batched_preds.append(
                {
                    name: (
                        np.array(np.array(preds[key]["boxes"]), dtype=np.float32)
                        if iou_type == "bbox"
                        else np.array(np.array(preds[key]["masks"]), dtype=np.uint8)
                    ),
                    "scores": np.array(preds[key]["scores"], dtype=np.float32),
                    "labels": np.array(preds[key]["labels"], dtype=np.int32),
                }
            )
            batched_target.append(
                {
                    name: (
                        np.array(target[key]["boxes"], dtype=np.float32)
                        if iou_type == "bbox"
                        else np.array(np.array(target[key]["masks"]), dtype=np.uint8)
                    ),
                    "labels": np.array(target[key]["labels"], dtype=np.int32),
                    "iscrowd": np.array(target[key]["iscrowd"], dtype=np.int32),
                    "area": np.array(target[key]["area"], dtype=np.float32),
                }
            )

        return batched_preds, batched_target

    def np_to_coco(self, name: str = "np_map_input") -> None:
        """Utility function for converting the input for this metric to coco format and saving it to a json file.

        This function should be used after calling `.update(...)` or `.forward(...)` on all data that should be written
        to the file, as the input is then internally cached. The function then converts to information to coco format
        a writes it to json files.

        Args:
            name: Name of the output file, which will be appended with "_preds.json" and "_target.json"

        Example:
            >>> import numpy as np
            >>> from metrics.detection import MeanAveragePrecision
            >>> preds = [
            ...   dict(
            ...     boxes=np.array([[258.0, 41.0, 606.0, 285.0]]),
            ...     scores=np.array([0.536]),
            ...     labels=np.array([0]),
            ...   )
            ... ]
            >>> target = [
            ...   dict(
            ...     boxes=np.array([[214.0, 41.0, 562.0, 285.0]]),
            ...     labels=np.array([0]),
            ...   )
            ... ]
            >>> metric = PrecisionRecallF1Support()
            >>> metric.update(preds, target)
            >>> metric.np_to_coco("np_map_input")  # doctest: +SKIP

        """
        target_dataset = self._get_coco_format(
            self.groundtruths, self.groundtruth_labels
        )
        preds_dataset = self._get_coco_format(
            self.detections, self.detection_labels, self.detection_scores
        )

        preds_json = json.dumps(preds_dataset["annotations"], indent=4)
        target_json = json.dumps(target_dataset, indent=4)

        with open(f"{name}_preds.json", "w") as f:
            f.write(preds_json)

        with open(f"{name}_target.json", "w") as f:
            f.write(target_json)

    def _get_safe_item_values(self, item: Dict[str, Any]) -> Union[np.ndarray, Tuple]:
        """Convert and return the boxes or masks from the item depending on the iou_type.

        Args:
            item: input dictionary containing the boxes or masks

        Returns:
            boxes or masks depending on the iou_type

        """
        if self.iou_type == "bbox":
            boxes = _fix_empty_arrays(item["boxes"])
            if boxes.size > 0:
                boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xywh")
            return boxes
        if self.iou_type == "segm":
            masks = []
            for i in item["masks"]:
                rle = mask_utils.encode(np.asfortranarray(i))
                masks.append((tuple(rle["size"]), rle["counts"]))
            return tuple(masks)
        raise Exception(f"IOU type {self.iou_type} is not supported")

    def _get_classes(self) -> List:
        """Return a list of unique classes found in ground truth and detection data."""
        all_labels = np.concatenate(self.detection_labels + self.groundtruth_labels)
        unique_classes = np.unique(all_labels)
        return unique_classes.tolist()

    def _get_coco_format(
        self,
        boxes: List[np.ndarray],
        labels: List[np.ndarray],
        scores: Optional[List[np.ndarray]] = None,
        crowds: Optional[List[np.ndarray]] = None,
        area: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """Transforms and returns all cached targets or predictions in COCO format.

        Format is defined at https://cocodataset.org/#format-data
        """
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for image_id, (image_boxes, image_labels) in enumerate(zip(boxes, labels)):
            if self.iou_type == "segm" and len(image_boxes) == 0:
                continue

            if self.iou_type == "bbox":
                image_boxes = image_boxes.tolist()
            image_labels = image_labels.tolist()

            images.append({"id": image_id})
            if self.iou_type == "segm":
                images[-1]["height"], images[-1]["width"] = (
                    image_boxes[0][0][0],
                    image_boxes[0][0][1],
                )

            for k, (image_box, image_label) in enumerate(
                zip(image_boxes, image_labels)
            ):
                if self.iou_type == "bbox" and len(image_box) != 4:
                    raise ValueError(
                        f"Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})"
                    )

                if not isinstance(image_label, int):
                    raise ValueError(
                        f"Invalid input class of sample {image_id}, element {k}"
                        f" (expected value of type integer, got type {type(image_label)})"
                    )

                stat = (
                    image_box
                    if self.iou_type == "bbox"
                    else {"size": image_box[0], "counts": image_box[1]}
                )

                if area is not None and area[image_id][k].tolist() > 0:
                    area_stat = area[image_id][k].tolist()
                else:
                    area_stat = (
                        image_box[2] * image_box[3]
                        if self.iou_type == "bbox"
                        else mask_utils.area(stat)
                    )

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox" if self.iou_type == "bbox" else "segmentation": stat,
                    "area": area_stat,
                    "category_id": image_label,
                    "iscrowd": (
                        crowds[image_id][k].tolist() if crowds is not None else 0
                    ),
                }

                if scores is not None:
                    score = scores[image_id][k].tolist()
                    if not isinstance(score, float):
                        raise ValueError(
                            f"Invalid input score of sample {image_id}, element {k}"
                            f" (expected value of type float, got type {type(score)})"
                        )
                    annotation["score"] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [{"id": i, "name": str(i)} for i in self._get_classes()]
        return {"images": images, "annotations": annotations, "categories": classes}
