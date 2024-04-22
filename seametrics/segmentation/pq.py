from typing import Set

import numpy as np
import torch
from torchmetrics.detection import PanopticQuality as PQ

class PanopticQuality():
    def __init__(self,
                 things: Set[int],
                 stuffs: Set[int]) -> None:
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
        self.metric = PQ(things=things, stuffs=stuffs)

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

        self.metric.update(preds, targets)

    def compute(self) -> torch.Tensor:
        """
        Computes the metric and returns the result.
        
        Returns:
            torch.Tensor: The computed metric result.
        """
        res = self.metric.compute()
        self.metric.reset()
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