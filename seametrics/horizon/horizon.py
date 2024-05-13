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
    height : int, optional
            The height of the camera in pixels. Defaults to 512.

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
                 vertical_fov_degrees=25.6,
                 height=512) -> None:
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
        height : int, optional
            The height of the camera in pixels. Defaults to 512.

        """
        self.vertical_fov_degrees = vertical_fov_degrees
        self.height = height
        self.slope_threshold = roll_to_slope(roll_threshold)
        self.midpoint_threshold = pitch_to_midpoint(pitch_threshold,
                                                    self.vertical_fov_degrees)

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

    def compute(self):
        """
        Compute the horizon error across the sequence.

        Returns
        -------
        float
            The computed horizon error.

        """

        # calculate erros and store values in slope_error_list and midpoint_error_list
        for annotated_horizon, proposed_horizon in zip(self.ground_truth_det,
                                                       self.predictions):

            if annotated_horizon is None or proposed_horizon is None:
                continue

            slope_error, midpoint_error = calculate_horizon_error(
                annotated_horizon, proposed_horizon)
            self.slope_error_list.append(slope_error)
            self.midpoint_error_list.append(midpoint_error)

        # calulcate detection rate
        detected_horizon_count = len(
            self.predictions) - self.predictions.count(None)
        detected_gt_count = len(
            self.ground_truth_det) - self.ground_truth_det.count(None)

        detection_rate = detected_horizon_count / detected_gt_count

        result = calculate_horizon_error_across_sequence(
            self.slope_error_list, self.midpoint_error_list,
            self.slope_threshold, self.midpoint_threshold,
            self.vertical_fov_degrees, self.height)

        result['detection_rate'] = detection_rate

        return result
