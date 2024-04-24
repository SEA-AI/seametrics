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
        self.device = self.select_device()
        self.metric = PQ(things=things, stuffs=stuffs, allow_unknown_preds_category=True)
        self.metric.to(self.device)


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

        preds, targets = preds.to(self.device), targets.to(self.device)    

        self.metric.update(preds, targets)
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