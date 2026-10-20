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

"""Precision, recall, F1 and support for object detection, in numpy.

Adapted from torchmetrics.detection.mean_ap.
"""

import contextlib
import io
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Literal

try:
    import pycocotools.mask as mask_utils
    from pycocotools.coco import COCO

    from seametrics.detection.cocoeval import COCOeval
except ImportError as exc:
    raise ModuleNotFoundError(
        "`MAP` metric requires that `pycocotools` installed."
        " Please install with `pip install pycocotools`"
    ) from exc

from seametrics.detection.np.utils import box_convert, fix_empty_arrays, input_validator

_AREA_RANGE_LEN = 2  # [min_area, max_area]
_BOX_LEN = 4


def _check_optional_lists(**args: Tuple[object, str]) -> None:
    """Raise ValueError unless each ``name=(value, item_kind)`` is None or a list."""
    for name, (value, kind) in args.items():
        if value is not None and not isinstance(value, list):
            raise ValueError(
                f"Expected argument `{name}` to be `None` or a list of {kind}"
                f" but got {value}"
            )


def _check_area_ranges(
    area_ranges: Optional[List[List[int]]], area_ranges_labels: Optional[List[str]]
) -> None:
    """Raise ValueError unless the area ranges and their labels are well formed."""
    if area_ranges is not None:
        if not isinstance(area_ranges, list):
            raise ValueError(
                "Expected argument `area_ranges` to be `None` or a list of lists"
                f" but got {area_ranges}"
            )
        if any(
            not isinstance(r, list) or len(r) != _AREA_RANGE_LEN for r in area_ranges
        ):
            raise ValueError(
                "Expected argument `area_ranges` to be a list of lists of length 2"
                f" but got {area_ranges}"
            )
    if area_ranges_labels is None:
        return
    if area_ranges is None:
        raise ValueError(
            "Expected argument `area_ranges_labels` to be `None`"
            " if `area_ranges` is not provided"
        )
    if not isinstance(area_ranges_labels, list):
        raise ValueError(
            "Expected argument `area_ranges_labels` to be `None` or a list of strings"
            f" but got {area_ranges_labels}"
        )
    if len(area_ranges_labels) != len(area_ranges):
        raise ValueError(
            "Expected argument `area_ranges_labels` to be a list of length"
            f" {len(area_ranges)} but got {area_ranges_labels}"
        )


# Holds one attribute per constructor argument plus the per-image state lists.
class PrecisionRecallF1Support:  # pylint: disable=too-many-instance-attributes
    r"""Compute the Precision, Recall, F1 and Support scores for object detection.

    - Precision = :math:`\frac{TP}{TP + FP}`
    - Recall = :math:`\frac{TP}{TP + FN}`
    - F1 = :math:`\frac{2 * Precision * Recall}{Precision + Recall}`
    - Support = :math:`TP + FN`

    As input to ``update`` the metric accepts:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing
      the key-values (each dictionary corresponds to a single image). Parameters
      that should be provided per dict:

        - boxes: (:class:`~np.ndarray`) of shape ``(num_boxes, 4)`` containing
          ``num_boxes`` detection boxes of the format specified in the
          constructor. By default, this method expects ``(xmin, ymin, xmax, ymax)``
          in absolute image coordinates.
        - scores: :class:`~np.ndarray` of shape ``(num_boxes)`` containing
          detection scores for the boxes.
        - labels: :class:`~np.ndarray` of shape ``(num_boxes)`` containing
          integer detection classes for the boxes.
        - masks: :class:`~np.ndarray` of shape
          ``(num_boxes, image_height, image_width)`` containing boolean masks.
          Only required when `iou_type="segm"`.

    - ``target`` (:class:`~List`) A list consisting of dictionaries each containing
      the key-values (each dictionary corresponds to a single image). Parameters
      that should be provided per dict:

        - boxes: :class:`~np.ndarray` of shape ``(num_boxes, 4)`` containing
          ``num_boxes`` ground truth boxes of the format specified in the
          constructor. By default, this method expects ``(xmin, ymin, xmax, ymax)``
          in absolute image coordinates.
        - labels: :class:`~np.ndarray` of shape ``(num_boxes)`` containing
          integer ground truth classes for the boxes.
        - masks: :class:`~np.ndarray` of shape
          ``(num_boxes, image_height, image_width)`` containing boolean masks.
          Only required when `iou_type="segm"`.
        - iscrowd: optional :class:`~np.ndarray` of shape ``(num_boxes)``, 1 where
          the box or mask covers a crowd of objects. Defaults to 0.
        - area: accepted for backwards compatibility but IGNORED. The area of
          every object is always computed from its bounding box (or mask), so
          area-range bucketing uses the same geometry as the IoU.

    ``compute`` returns a dict with:

    - ``params``: the COCOeval ``Params`` object
    - ``eval``: output of ``COCOeval.accumulate()``
    - ``metrics``: per area-range label, at the lowest IoU threshold and the
      largest ``maxDets``: ``range``, ``iouThr`` (str), ``maxDets``, ``tp``,
      ``fp``, ``fn``, ``duplicates``, ``precision``, ``recall``, ``f1``,
      ``support`` (tp + fn), ``fpi`` and ``nImgs``. Values are scalars when
      ``class_agnostic``, else per-class arrays; undefined scores are -1.

    .. note::
        This metric uses a modified copy of the `pycocotools` COCOeval as its
        backend, so it requires `pycocotools` to be installed.

    Args:
        box_format:
            Input format of given boxes. Supported formats are
            ``[xyxy, xywh, cxcywh]``.
        iou_type:
            Type of input (either masks or bounding-boxes) used for computing IOU.
            Supported IOU types are ``["bbox", "segm"]``. If using ``"segm"``,
            masks should be provided in input.
        iou_thresholds:
            IoU thresholds for evaluation. If set to ``None`` it corresponds to the
            stepped range ``[0.5,...,0.95]`` with step ``0.05``. Else provide a
            list of floats.
        rec_thresholds:
            Recall thresholds for evaluation. If set to ``None`` it corresponds to
            the stepped range ``[0,...,1]`` with step ``0.01``. Else provide a list
            of floats.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use
            thresholds ``[100]``. Else, please provide a list of ints.
        area_ranges:
            Area ranges for evaluation. If set to ``None`` it corresponds to the
            ranges ``[[0^2, 1e5^2]]``. Else, please provide a list of lists of
            length 2.
        area_ranges_labels:
            Labels for the area ranges. If set to ``None`` it corresponds to the
            labels ``["all"]``. Else, please provide a list of strings of the same
            length as ``area_ranges``.
        class_agnostic:
            If ``True`` will compute metrics globally. If ``False`` will compute
            metrics per class. Default: ``True``.
        debug:
            If ``True`` will print the COCO indexing and evaluation log to stdout.
        labels:
            Labels to evaluate when ``class_agnostic`` is ``False``. If set to
            ``None``, every label seen in the inputs is used.
        _kwargs: Accepted for torchmetrics API compatibility and ignored.

    Raises:
        ValueError: If ``box_format`` or ``iou_type`` is unsupported; if
            ``iou_thresholds``, ``rec_thresholds``, ``max_detection_thresholds``,
            ``area_ranges``, ``area_ranges_labels`` or ``labels`` is neither None
            nor a list; if an area range is not a list of length 2; if
            ``area_ranges_labels`` is given without ``area_ranges`` or differs
            from it in length; if ``class_agnostic`` or ``debug`` is not a bool;
            or if ``labels`` is given while ``class_agnostic`` is True.

    Example:
        >>> import numpy as np
        >>> from seametrics.detection import PrecisionRecallF1Support
        >>> preds = [
        ...     dict(
        ...         boxes=np.array([[258.0, 41.0, 606.0, 285.0]]),
        ...         scores=np.array([0.536]),
        ...         labels=np.array([0]),
        ...     )
        ... ]
        >>> target = [
        ...     dict(
        ...         boxes=np.array([[214.0, 41.0, 562.0, 285.0]]),
        ...         labels=np.array([0]),
        ...     )
        ... ]
        >>> metric = PrecisionRecallF1Support()
        >>> metric.update(preds, target)
        >>> print(metric.compute())
        {'params': <seametrics.detection.cocoeval.Params object at 0x...>,
         'eval': {...},
         'metrics': {'all': {'range': [0, 10000000000.0],
         'iouThr': '0.50',
         'maxDets': 100,
         'tp': 1,
         'fp': 0,
         'fn': 0,
         'duplicates': 0,
         'precision': 1.0,
         'recall': 1.0,
         'f1': 1.0,
         'support': 1,
         'fpi': 0,
         'nImgs': 1}}}
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

    # Public constructor: every argument is part of the API and stored on the instance.
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        labels: Optional[List[int]] = None,
        **_kwargs: object,
    ) -> None:
        """Initializes the metric; the arguments are documented on the class."""
        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats}"
                f" but got {box_format}"
            )
        self.box_format = box_format

        allowed_iou_types = ("segm", "bbox")
        if iou_type not in allowed_iou_types:
            raise ValueError(
                f"Expected argument `iou_type` to be one of {allowed_iou_types}"
                f" but got {iou_type}"
            )
        self.iou_type = iou_type

        _check_optional_lists(
            iou_thresholds=(iou_thresholds, "floats"),
            rec_thresholds=(rec_thresholds, "floats"),
            max_detection_thresholds=(max_detection_thresholds, "ints"),
        )
        self.iou_thresholds = (
            iou_thresholds
            or np.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()
        )
        self.rec_thresholds = (
            rec_thresholds or np.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()
        )
        max_det_thr = np.sort(
            np.array(max_detection_thresholds or [100], dtype=np.uint)
        )
        self.max_detection_thresholds = max_det_thr.tolist()

        _check_area_ranges(area_ranges, area_ranges_labels)
        self.area_ranges = area_ranges if area_ranges is not None else [[0**2, 1e5**2]]
        self.area_ranges_labels = (
            area_ranges_labels if area_ranges_labels is not None else ["all"]
        )

        if not isinstance(class_agnostic, bool):
            raise ValueError("Expected argument `class_agnostic` to be a boolean")
        self.class_agnostic = class_agnostic

        if labels is not None and not isinstance(labels, list):
            raise ValueError(
                f"Expected argument `labels` to be a list of integers, but got {labels}"
            )
        self.labels = labels

        if self.labels is not None and self.class_agnostic:
            raise ValueError(
                "Expected labels to be None if argument `class_agnostic` is True."
            )

        if not isinstance(debug, bool):
            raise ValueError("Expected argument `debug` to be a boolean")
        self.debug = debug

        self.detections = []
        self.detection_scores = []
        self.detection_labels = []
        self.groundtruths = []
        self.groundtruth_labels = []
        self.groundtruth_crowds = []

    def update(
        self, preds: List[Dict[str, np.ndarray]], target: List[Dict[str, np.ndarray]]
    ) -> None:
        """Update metric state.

        ``input_validator`` checks the inputs. Malformed boxes, labels or scores
        raise ValueError later, in ``compute``.
        """
        input_validator(preds, target, iou_type=self.iou_type, labels=self.labels)

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

    def compute(self) -> dict:
        """Computes the metric."""
        coco_target, coco_preds = COCO(), COCO()

        coco_target.dataset = self._get_coco_format(
            self.groundtruths,
            self.groundtruth_labels,
            crowds=self.groundtruth_crowds,
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
            if self.class_agnostic:
                coco_eval.params.useCats = 0
            else:
                coco_eval.params.useCats = 1
                if not self.labels:
                    all_labels = np.unique(
                        np.concatenate(self.detection_labels).tolist()
                        + np.concatenate(self.groundtruth_labels).tolist()
                    ).tolist()
                else:
                    all_labels = self.labels
                coco_eval.params.catIds = all_labels
            coco_eval.evaluate()
            coco_eval.accumulate()

        if self.debug:
            print(f.getvalue())

        metrics = coco_eval.summarize()
        return metrics

    @staticmethod
    def coco_to_np(
        coco_preds: str,
        coco_target: str,
        iou_type: Literal["bbox", "segm"] = "bbox",
    ) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
        """Converts .json coco format files to the input format of this metric.

        Each image becomes a dict of ``boxes`` (COCO ``xywh``) or ``masks``,
        ``labels`` and, for predictions, ``scores``; targets also carry
        ``iscrowd`` and ``area``.

        Args:
            coco_preds: Path to the json file containing the predictions in coco
                format
            coco_target: Path to the json file containing the targets in coco format
            iou_type: Type of input, either `bbox` for bounding boxes or `segm` for
                segmentation masks

        Returns:
            preds: List of dictionaries containing the predictions in the input
                format of this metric
            target: List of dictionaries containing the targets in the input format
                of this metric

        Example:
            >>> # File formats are defined at https://cocodataset.org/#format-data
            >>> # Example files can be found at
            >>> # https://github.com/cocodataset/cocoapi/tree/master/results
            >>> preds, target = PrecisionRecallF1Support.coco_to_np(
            ...     "preds.json", "target.json", iou_type="bbox"
            ... )  # doctest: +SKIP

        """
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(coco_target)
            dt = gt.loadRes(coco_preds)

        target = {}
        for t in gt.dataset["annotations"]:
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
        for p in dt.dataset["annotations"]:
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
        name = "boxes" if iou_type == "bbox" else "masks"
        for key, tgt in target.items():
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
                        np.array(tgt["boxes"], dtype=np.float32)
                        if iou_type == "bbox"
                        else np.array(np.array(tgt["masks"]), dtype=np.uint8)
                    ),
                    "labels": np.array(tgt["labels"], dtype=np.int32),
                    "iscrowd": np.array(tgt["iscrowd"], dtype=np.int32),
                    "area": np.array(tgt["area"], dtype=np.float32),
                }
            )

        return batched_preds, batched_target

    def np_to_coco(self, name: str = "np_map_input") -> None:
        """Converts the input for this metric to coco format and saves it to json.

        Call after the last ``update``: converts the cached inputs to coco format
        and writes ``{name}_preds.json`` and ``{name}_target.json``.

        Args:
            name: Name of the output file, which will be appended with "_preds.json"
                and "_target.json"

        Example:
            >>> import numpy as np
            >>> from seametrics.detection import PrecisionRecallF1Support
            >>> preds = [
            ...     dict(
            ...         boxes=np.array([[258.0, 41.0, 606.0, 285.0]]),
            ...         scores=np.array([0.536]),
            ...         labels=np.array([0]),
            ...     )
            ... ]
            >>> target = [
            ...     dict(
            ...         boxes=np.array([[214.0, 41.0, 562.0, 285.0]]),
            ...         labels=np.array([0]),
            ...     )
            ... ]
            >>> metric = PrecisionRecallF1Support()
            >>> metric.update(preds, target)
            >>> metric.np_to_coco("np_map_input")  # doctest: +SKIP

        """
        target_dataset = self._get_coco_format(
            self.groundtruths, self.groundtruth_labels
        )
        preds_dataset = self._get_coco_format(
            self.detections, self.detection_labels, scores=self.detection_scores
        )

        preds_json = json.dumps(preds_dataset["annotations"], indent=4)
        target_json = json.dumps(target_dataset, indent=4)

        with pathlib.Path(f"{name}_preds.json").open("w", encoding="utf-8") as f:
            f.write(preds_json)

        with pathlib.Path(f"{name}_target.json").open("w", encoding="utf-8") as f:
            f.write(target_json)

    def _get_safe_item_values(self, item: Dict[str, Any]) -> Union[np.ndarray, Tuple]:
        """Return the item's boxes as ``xywh``, or its masks as RLE (``iou_type``)."""
        if self.iou_type == "bbox":
            boxes = fix_empty_arrays(item["boxes"])
            if boxes.size > 0:
                boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xywh")
            return boxes
        if self.iou_type == "segm":
            masks = []
            for i in item["masks"]:
                rle = mask_utils.encode(np.asfortranarray(i))
                masks.append((tuple(rle["size"]), rle["counts"]))
            return tuple(masks)
        raise ValueError(f"IOU type {self.iou_type} is not supported")

    def _get_classes(self) -> List:
        """Return a list of unique classes found in ground truth and detection data."""
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            all_labels = np.concatenate(self.detection_labels + self.groundtruth_labels)
            return np.unique(all_labels).tolist()
        return []

    def _get_coco_format(
        self,
        boxes: List[np.ndarray],
        labels: List[np.ndarray],
        *,
        scores: Optional[List[np.ndarray]] = None,
        crowds: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """Transforms and returns all cached targets or predictions in COCO format.

        Format is defined at https://cocodataset.org/#format-data

        The ``area`` of every annotation is always derived from the bounding box
        (or the mask, for ``iou_type="segm"``). Any ``area`` supplied by the
        caller is ignored, so area-range bucketing depends only on the geometry
        that is also used for the IoU.
        """
        images = []
        annotations = []

        for image_id, (image_boxes, image_labels) in enumerate(
            zip(boxes, labels, strict=False)
        ):
            if self.iou_type == "segm" and len(image_boxes) == 0:
                continue

            images.append({"id": image_id})
            if self.iou_type == "segm":
                images[-1]["height"], images[-1]["width"] = (
                    image_boxes[0][0][0],
                    image_boxes[0][0][1],
                )

            for k, (image_box, image_label) in enumerate(
                zip(
                    image_boxes.tolist() if self.iou_type == "bbox" else image_boxes,
                    image_labels.tolist(),
                    strict=False,
                )
            ):
                annotations.append(
                    self._get_coco_annotation(
                        (image_id, k),
                        image_box,
                        image_label,
                        # ids have to start with 1, otherwise COCOEval results are wrong
                        annotation_id=len(annotations) + 1,
                        crowd=0 if crowds is None else crowds[image_id][k].tolist(),
                        score=None if scores is None else scores[image_id][k].tolist(),
                    )
                )

        classes = [{"id": i, "name": str(i)} for i in self._get_classes()]
        return {"images": images, "annotations": annotations, "categories": classes}

    def _get_coco_annotation(
        self,
        index: Tuple[int, int],
        image_box: Union[List[float], Tuple],
        image_label: object,
        *,
        annotation_id: int,
        crowd: int,
        score: Optional[object],
    ) -> Dict:
        """Validates one box and returns it as a COCO annotation.

        Args:
            index: ``(image_id, k)``, the image and the box's position in it.
            image_box: ``xywh`` box, or ``(size, counts)`` RLE mask for ``segm``.
            image_label: Class of the box; must be an int.
            annotation_id: Unique, 1-based id of the annotation.
            crowd: The ``iscrowd`` flag.
            score: Detection score, must be a float; ``None`` for ground truth.

        Raises:
            ValueError: If the box, label or score is malformed.
        """
        image_id, k = index
        if self.iou_type == "bbox" and len(image_box) != _BOX_LEN:
            raise ValueError(
                f"Invalid input box of sample {image_id}, element {k}"
                f" (expected 4 values, got {len(image_box)})"
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

        area = (
            image_box[2] * image_box[3]
            if self.iou_type == "bbox"
            else mask_utils.area(stat)
        )

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "bbox" if self.iou_type == "bbox" else "segmentation": stat,
            "area": area,
            "category_id": image_label,
            "iscrowd": crowd,
        }

        if score is not None:
            if not isinstance(score, float):
                raise ValueError(
                    f"Invalid input score of sample {image_id}, element {k}"
                    f" (expected value of type float, got type {type(score)})"
                )
            annotation["score"] = score
        return annotation
