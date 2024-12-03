from seametrics.user_friendly.utils import calculate_from_payload


class UserFriendlyMetrics:
    """
    Class for computing UserFriendly metrics.

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    update(predictions, ground_truth_det)
        Update the predictions and ground truth detections.
    compute()
        Compute the UserFriendly metrics across the sequence.

    """

    def __init__(
        self, payload, max_iou, filters, recognition_thresholds, debug
    ) -> None:
        """
        Initialize the UserFriendly class.

        Parameters
        ----------

        """
        self.payload = payload
        self.max_iou = max_iou
        self.filters = filters
        self.recognition_thresholds = recognition_thresholds
        self.debug = debug

    def update(self, predictions, ground_truth_det) -> None:
        """
        Update the predictions and ground truth detections.

        Parameters
        ----------.

        """

        print("user-friendly-metrics updated")

    def _compute(self):
        """Returns the scores"""
        # TODO: Compute the different scores of the module
        return calculate_from_payload(
            self.payload,
            self.max_iou,
            self.filters,
            self.recognition_thresholds,
            self.debug,
        )
