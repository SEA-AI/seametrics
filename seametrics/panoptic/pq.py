from typing import Set, List, Tuple, Collection, Any

import numpy as np
import torch
from torchmetrics.detection import PanopticQuality as PQ
from torchmetrics.functional.detection._panoptic_quality_common import (
    _prepocess_inputs,
    _validate_inputs,
)

from seametrics.panoptic.tm.functionalities import (
    _panoptic_quality_compute,
    _panoptic_quality_update
)

class AreaPanopticQuality(PQ):
    def __init__(self,
                 things: Collection[int],
                 stuffs: Collection[int],
                 areas: List[Tuple[float]] = [(0, 1e10)],
                 allow_unknown_preds_category: bool = False,
                 return_sq_and_rq: bool = False,
                 return_per_class: bool = False,
                 **kwargs: Any):
        super().__init__(
            things=things, 
            stuffs=stuffs, 
            allow_unknown_preds_category=allow_unknown_preds_category, 
            return_sq_and_rq=return_sq_and_rq, 
            return_per_class=return_per_class, 
            **kwargs
        )
        self.areas = areas
        num_categories = len(things) + len(stuffs)
        self.add_state("iou_sum", default=torch.zeros(len(areas), num_categories, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("true_positives", default=torch.zeros(len(areas), num_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(len(areas), num_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(len(areas), num_categories, dtype=torch.int), dist_reduce_fx="sum")

    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        r"""Update state with predictions and targets.

        Args:
            preds: panoptic detection of shape ``[batch, *spatial_dims, 2]`` containing
                the pair ``(category_id, instance_id)`` for each point.
                If the ``category_id`` refer to a stuff, the instance_id is ignored.

            target: ground truth of shape ``[batch, *spatial_dims, 2]`` containing
                the pair ``(category_id, instance_id)`` for each pixel of the image.
                If the ``category_id`` refer to a stuff, the instance_id is ignored.

        Raises:
            TypeError:
                If ``preds`` or ``target`` is not an ``torch.Tensor``.
            ValueError:
                If ``preds`` and ``target`` have different shape.
            ValueError:
                If ``preds`` has less than 3 dimensions.
            ValueError:
                If the final dimension of ``preds`` has size != 2.

        """
        _validate_inputs(preds, target)
        flatten_preds = _prepocess_inputs(
            self.things, self.stuffs, preds, self.void_color, self.allow_unknown_preds_category
        )
        flatten_target = _prepocess_inputs(self.things, self.stuffs, target, self.void_color, True)
        iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
            flatten_preds, flatten_target, self.cat_id_to_continuous_id, self.void_color
        )
        self.iou_sum += iou_sum
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives

    def compute(self) -> torch.Tensor:
        """Compute panoptic quality based on inputs passed in to ``update`` previously."""
        pq, sq, rq, pq_avg, sq_avg, rq_avg = _panoptic_quality_compute(
            self.iou_sum, self.true_positives, self.false_positives, self.false_negatives
        )
        if self.return_per_class:
            if self.return_sq_and_rq:
                return torch.stack((pq, sq, rq), dim=-1)
            return pq.view(1, -1)
        if self.return_sq_and_rq:
            return torch.stack((pq_avg, sq_avg, rq_avg), dim=0)
        return pq_avg

class PanopticQuality():
    def __init__(self,
            things: Set[int],
            stuffs: Set[int],
            areas: List[Tuple[float]] = [(0, 1e10)],
            return_sq_and_rq: bool = True,
            return_per_class: bool = True,
            CHUNK_SIZE: int = 200
        ) -> None:
        """
        Initializes the PanopticQuality class with the given sets of things and stuffs.

        Parameters:
            things (Set[int]): A set of integers representing the things.
            stuffs (Set[int]): A set of integers representing the stuffs.

        Returns:
            None
        """
        self.things = things
        self.stuffs = stuffs
        self.device = self.select_device()
        self.metric = AreaPanopticQuality(
            things=things,
            stuffs=stuffs,
            allow_unknown_preds_category=True,
            return_sq_and_rq=return_sq_and_rq,
            return_per_class=return_per_class
        )
        self.metric.to(self.device)
        self.CHUNK_SIZE = CHUNK_SIZE

    @staticmethod
    def select_device():
        # Check for CUDA GPU availability
        if torch.cuda.is_available():
            return torch.device('cuda')
        
        # Check for MPS availability (for macOS on Apple Silicon)
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        
        # Default to CPU if neither CUDA nor MPS is available
        else:
            return torch.device('cpu')

    def update(self,
               preds: torch.Tensor,
               targets: torch.Tensor) -> None:
        """
        Updates the metric with the given predictions and targets.
        Note: instance ids are ignored for label categories belonging to stuffs

        Parameters:
            preds (torch.Tensor): A tensor of shape (batch_size, img_height, img_width, 2) representing the predictions.
                                   The last dimension contains the label category at index 0 and the instance id at index 1.
            targets (torch.Tensor): A tensor of shape (batch_size, img_height, img_width, 2) representing the targets.
                                     The last dimension contains the label category at index 0 and the instance id at index 1.

        Returns:
            None
        """
        if type(preds) == np.ndarray:
            preds = torch.from_numpy(preds)
        if type(targets) == np.ndarray:
            targets = torch.from_numpy(targets)

        for pred_chunk, target_chunk in zip(torch.split(preds, self.CHUNK_SIZE), torch.split(targets, self.CHUNK_SIZE)):
            pred_chunk, target_chunk = pred_chunk.to(self.device), target_chunk.to(self.device)    
            self.metric.update(pred_chunk, target_chunk)
            pred_chunk.to("cpu"), target_chunk.to("cpu")

        print("Added data ...")

    def compute(self) -> torch.Tensor:
        """
        Computes the metric and returns the result.
        
        Returns:
            torch.Tensor: The computed metric result.
        """
        print("Start computing ...")
        res = self.metric.compute()
        self.metric.reset()
        print("Finished!")
        return res
    
    def update_and_compute(self,
                           preds: torch.Tensor, 
                           targets: torch.Tensor) -> torch.Tensor:
        """
        Updates the metric with the given predictions and targets.
        For more info about parameters, see docstring of self.update().
        
        Parameters:
            preds (torch.Tensor): A tensor of shape (batch_size, img_height, img_width, 2) representing the predictions.
            targets (torch.Tensor): A tensor of shape (batch_size, img_height, img_width, 2) representing the targets.
        
        Returns:
            torch.Tensor: The computed metric result.
        """
        self.update(preds, targets)
        return self.compute()