from seametrics.horizon.utils import *


class HorizonMetrics:
    """
    Class for computing horizon metrics.

    Parameters
    ----------
    roll_threshold : float, optional
        The roll threshold in radians. Defaults to 0.5.
    pitch_threshold : float, optional
        The pitch threshold in radians. Defaults to 0.1.
    vertical_fov_degrees : float, optional
        The vertical field of view of the camera in degrees. Defaults to 25.6.

    Attributes
    ----------
    slope_threshold : float
        The slope threshold calculated from the roll threshold.
    midpoint_threshold : float
        The midpoint threshold calculated from the pitch threshold and vertical field of view.

    Methods
    -------
    update(predictions, ground_truth_det)
        Update the predictions and ground truth detections.
    compute()
        Compute the horizon error across the sequence.

    """

    def __init__(self,
                 roll_threshold=0.5,
                 pitch_threshold=0.1,
                 vertical_fov_degrees=25.6) -> None:
        """
        Initialize the HorizonMetrics class.

        Parameters
        ----------
        roll_threshold : float, optional
            The roll threshold in radians. Defaults to 0.5.
        pitch_threshold : float, optional
            The pitch threshold in radians. Defaults to 0.1.
        vertical_fov_degrees : float, optional
            The vertical field of view of the camera in degrees. Defaults to 25.6.

        """
        self.ververtical_fov_degrees = vertical_fov_degrees
        self.slope_threshold = roll_to_slope(roll_threshold)
        self.midpoint_threshold = pitch_to_midpoint(pitch_threshold,
                                                    vertical_fov_degrees)

    def update(self, predictions, ground_truth_det) -> None:
        """
        Update the predictions and ground truth detections.

        Parameters
        ----------
        predictions : list
            List of predicted horizons.
        ground_truth_det : list
            List of ground truth horizons.

        """
        self.predictions = predictions
        self.ground_truth_det = ground_truth_det
        self.slope_error_list = []
        self.midpoint_error_list = []

        for annotated_horizon, proposed_horizon in zip(self.ground_truth_det,
                                                       self.predictions):
            slope_error, midpoint_error = calculate_horizon_error(
                annotated_horizon, proposed_horizon)
            self.slope_error_list.append(slope_error)
            self.midpoint_error_list.append(midpoint_error)

    def compute(self):
        """
        Compute the horizon error across the sequence.

        Returns
        -------
        float
            The computed horizon error.

        """
        return calculate_horizon_error_across_sequence(
            self.slope_error_list, self.midpoint_error_list,
            self.slope_threshold, self.midpoint_threshold)
