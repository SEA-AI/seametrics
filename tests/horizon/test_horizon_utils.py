from seametrics.horizon.utils import (xy_points_to_slope_midpoint,
                                      calculate_horizon_error, slope_to_roll,
                                      roll_to_slope, midpoint_to_pitch,
                                      pitch_to_midpoint,
                                      calculate_horizon_error_across_sequence)
import pytest
import numpy as np


def test_xy_points_to_slope_midpoint():
    xy_point = [[0.0, 0.0], [1.0, 1.0]]
    slope, midpoint = xy_points_to_slope_midpoint(xy_point)

    assert slope == 1.0
    assert midpoint == [0.5, 0.5]

    xy_point = [[0.0, 0.0], [1.0, 0.5]]
    slope, midpoint = xy_points_to_slope_midpoint(xy_point)

    assert slope == 0.5
    assert midpoint == [0.5, 0.25]


def test_calculate_horizon_error():
    horizon_gt = [[0.0, 0.0], [1.0, 1.0]]
    horizon_pred = [[0.0, 0.0], [1.0, 0.5]]

    slope_gt = 1.0
    slope_pred = 0.5

    midpoint_gt = [0.5, 0.5]
    midpoint_pred = [0.5, 0.25]

    slope_error, midpoint_error = calculate_horizon_error(
        annotated_horizon=horizon_gt, proposed_horizon=horizon_pred)

    assert slope_error == abs(slope_gt - slope_pred)
    assert midpoint_error == abs(midpoint_gt[1] - midpoint_pred[1])


def test_slope_and_roll():
    slope = 1.0
    roll = slope_to_roll(slope)
    assert roll == 45.0

    roll = 45.0
    slope = roll_to_slope(roll)
    assert slope == pytest.approx(1.0)


def test_midpoint_and_pitch():
    pitch = 12.8
    midpoint = pitch_to_midpoint(pitch, 25.6)

    assert midpoint == 0.5

    midpoint = 0.5
    pitch = midpoint_to_pitch(midpoint, 25.6)
    assert pitch == 12.8


def test_calculate_horizon_across_sequences():
    slope_error_list = [0.5, 0.25]
    midpoint_error_list = [0.25, 0.125]
    height = 512
    vertical_fov_degrees = 25.6
    slope_error_jump_threshold = 0.1
    midpoint_error_jump_threshold = 0.1
    result = calculate_horizon_error_across_sequence(
        slope_error_list=slope_error_list,
        midpoint_error_list=midpoint_error_list,
        slope_error_jump_threshold=slope_error_jump_threshold,
        midpoint_error_jump_threshold=midpoint_error_jump_threshold,
        vertical_fov_degrees=vertical_fov_degrees,
        height=height)

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
