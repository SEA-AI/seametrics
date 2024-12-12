from seametrics.user_friendly.utils import calculate_from_payload


class UserFriendlyMetrics:
    """
    Class for computing UserFriendly metrics.

    Methods
    -------
    compute()
        Compute the UserFriendly metrics.

    """

    def __init__(
        self, payload, max_iou, filters, recognition_thresholds, debug
    ) -> None:
        """
        Initialize the UserFriendly class.

        Parameters
        ----------
        payload : Payload
            The payload object.
        max_iou : float
            The maximum intersection over union (IoU) threshold.
        filters : dict
            A dictionary of filters to apply to the data.
        recognition_thresholds : list
            A list of recognition thresholds to use.
        debug : bool
            Whether to print debug messages.

        """
        self.payload = payload
        self.max_iou = max_iou
        self.filters = filters
        self.recognition_thresholds = recognition_thresholds
        self.debug = debug

    def _compute(self):
        """Returns the scores"""
        return calculate_from_payload(
            self.payload,
            self.max_iou,
            self.filters,
            self.recognition_thresholds,
            self.debug,
        )
