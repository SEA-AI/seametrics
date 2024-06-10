from seametrics.horizon.horizon import HorizonMetrics
from seametrics.horizon.utils import calculate_horizon_error_across_sequence, calculate_horizon_error
import numpy as np


def get_fake_data(no_errors=False):
    ground_truth_points = [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 0.5]],
                           [[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 0.5]]]

    prediction_points = [[[0.0, 0.0], [1.0, 1.5]], [[0.0, 0.0], [1.0, 0.25]],
                         None, None]
    if no_errors:
        prediction_points = ground_truth_points

    return ground_truth_points, prediction_points


def test_horizon_metrics_no_errors():
    ground_truth_points, prediction_points = get_fake_data(no_errors=True)
    # Create the metrics object

    metrics = HorizonMetrics(roll_threshold=0.5,
                             pitch_threshold=0.1,
                             vertical_fov_degrees=25.6,
                             height=512)

    # Set ground truth and predictions
    metrics.update(predictions=prediction_points,
                   ground_truth_det=ground_truth_points)

    # Compute metrics
    result = metrics.compute()
    assert result['average_slope_error'] == 0.0
    assert result['average_midpoint_error'] == 0.0
    assert result['average_midpoint_error_px'] == 0.0
    assert result['stddev_slope_error'] == 0.0
    assert result['stddev_midpoint_error'] == 0.0
    assert result['stddev_midpoint_error_px'] == 0.0
    assert result['max_slope_error'] == 0.0
    assert result['max_midpoint_error'] == 0.0
    assert result['max_midpoint_error_px'] == 0.0
    assert result['num_slope_error_jumps'] == 0
    assert result['num_midpoint_error_jumps'] == 0
    assert result['average_slope_error'] == 0.0
    assert result['detection_rate'] == 1.0


def test_horizon_metrics_errors():
    ground_truth_points, prediction_points = get_fake_data(no_errors=False)
    # Create the metrics object

    height = 512
    vertical_fov_degrees = 25.6
    roll_threshold = 0.5
    pitch_threshold = 0.1

    # Create the metrics object
    metrics = HorizonMetrics(roll_threshold=roll_threshold,
                             pitch_threshold=pitch_threshold,
                             vertical_fov_degrees=vertical_fov_degrees,
                             height=height)

    # Set ground truth and predictions
    metrics.update(predictions=prediction_points,
                   ground_truth_det=ground_truth_points)

    # Compute metrics
    result = metrics.compute()

    slope_error_list = []
    midpoint_error_list = []

    for point_gt, point_pred in zip(ground_truth_points, prediction_points):
        if point_gt is None or point_pred is None:
            continue

        slope_error, midpoint_error = calculate_horizon_error(
            point_gt, point_pred)
        slope_error_list.append(slope_error)
        midpoint_error_list.append(midpoint_error)

    average_midpoint_error = np.mean(midpoint_error_list)
    average_midpoint_error_px = average_midpoint_error * height
    average_midpoint_error_deg = average_midpoint_error * vertical_fov_degrees

    std_midpoint_error = np.std(midpoint_error_list)
    std_midpoint_error_px = std_midpoint_error * height
    std_midpoint_error_deg = std_midpoint_error * vertical_fov_degrees

    max_midpoint_error = np.max(midpoint_error_list)
    max_midpoint_error_px = max_midpoint_error * height
    max_midpoint_error_deg = max_midpoint_error * vertical_fov_degrees

    average_slope_error = np.mean(slope_error_list)
    average_slope_error_deg = np.arctan(average_slope_error) * 180 / np.pi

    std_slope_error = np.std(slope_error_list)
    std_slope_error_deg = np.arctan(std_slope_error) * 180 / np.pi

    max_slope_error = np.max(slope_error_list)
    max_slope_error_deg = np.arctan(max_slope_error) * 180 / np.pi

    assert result['average_midpoint_error'] == average_midpoint_error_deg
    assert result['average_midpoint_error_px'] == average_midpoint_error_px

    assert result['stddev_midpoint_error'] == std_midpoint_error_deg
    assert result['stddev_midpoint_error_px'] == std_midpoint_error_px

    assert result['max_midpoint_error'] == max_midpoint_error_deg
    assert result['max_midpoint_error_px'] == max_midpoint_error_px

    assert result['average_slope_error'] == average_slope_error_deg
    assert result['stddev_slope_error'] == std_slope_error_deg
    assert result['max_slope_error'] == max_slope_error_deg

    assert result['num_midpoint_error_jumps'] == 1
    assert result['num_slope_error_jumps'] == 1

    assert result['detection_rate'] == 0.5
