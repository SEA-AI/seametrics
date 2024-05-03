import numpy as np


def xy_points_to_slope_midpoint(xy_points):
    """
    Given two points, return the slope and midpoint of the line

    Args:
    xy_points: list of two points, each point is a list of two elements
    Points are in the form of [x, y], where x and y are normalized to [0, 1]

    Returns:
    slope: Slope of the line
    midpoint : Midpoint is in the form of [x,y], and is also normalized to [0, 1]
    """

    x1, y1, x2, y2 = xy_points[0][0], xy_points[0][1], xy_points[1][
        0], xy_points[1][1]
    slope = (y2 - y1) / (x2 - x1)

    midpoint_x = 0.5
    midpoint_y = slope * (0.5 - x1) + y1
    midpoint = [midpoint_x, midpoint_y]
    return slope, midpoint


def calculate_horizon_error(annotated_horizon, proposed_horizon):
    """
    Calculate the error between the annotated horizon and the proposed horizon

    Args:
    annotated_horizon: list of two points, each point is a list of two elements
    Points are in the form of [x, y], where x and y are normalized to [0, 1]
    proposed_horizon: list of two points, each point is a list of two elements
    Points are in the form of [x, y], where x and y are normalized to [0, 1]

    Returns:
    slope_error: Error in the slope of the lines
    midpoint_error: Error in the midpoint_y of the lines
    """

    slope_annotated, midpoint_annotated = xy_points_to_slope_midpoint(
        annotated_horizon)
    slope_proposed, midpoint_proposed = xy_points_to_slope_midpoint(
        proposed_horizon)

    slope_error = abs(slope_annotated - slope_proposed)
    midpoint_error = abs(midpoint_annotated[1] - midpoint_proposed[1])

    return slope_error, midpoint_error


def calculate_horizon_error_across_sequence(slope_error_list,
                                            midpoint_error_list,
                                            slope_error_jump_threshold,
                                            midpoint_error_jump_threshold):
    """
    Calculate the error statistics across a sequence of frames

    Args:
    slope_error_list: List of errors in the slope of the lines
    midpoint_error_list: List of errors in the midpoint_y of the lines

    Returns:
    average_slope_error: Average error in the slope of the lines
    average_midpoint_error: Average error in the midpoint_y of the lines
    """

    # Calculate the average and standard deviation of the errors
    average_slope_error = np.mean(slope_error_list)
    average_midpoint_error = np.mean(midpoint_error_list)

    stddev_slope_error = np.std(slope_error_list)
    stddev_midpoint_error = np.std(midpoint_error_list)

    # Calculate the maximum errors
    max_slope_error = np.max(slope_error_list)
    max_midpoint_error = np.max(midpoint_error_list)

    # Calculate the differences between errors in successive frames
    diff_slope_error = np.abs(np.diff(slope_error_list))
    diff_midpoint_error = np.abs(np.diff(midpoint_error_list))

    # Calculate the number of jumps in the errors
    num_slope_error_jumps = np.sum(
        diff_slope_error > slope_error_jump_threshold)
    num_midpoint_error_jumps = np.sum(
        diff_midpoint_error > midpoint_error_jump_threshold)

    # Create a dictionary to store the results
    sequence_results = {
        'average_slope_error': average_slope_error,
        'average_midpoint_error': average_midpoint_error,
        'stddev_slope_error': stddev_slope_error,
        'stddev_midpoint_error': stddev_midpoint_error,
        'max_slope_error': max_slope_error,
        'max_midpoint_error': max_midpoint_error,
        'num_slope_error_jumps': num_slope_error_jumps,
        'num_midpoint_error_jumps': num_midpoint_error_jumps
    }

    return sequence_results


import numpy as np
import cv2
import matplotlib.pyplot as plt


def xy_points_to_slope_midpoint(xy_points):
    """
    Given two points, return the slope and midpoint of the line

    Args:
    xy_points: list of two points, each point is a list of two elements
    Points are in the form of [x, y], where x and y are normalized to [0, 1]

    Returns:
    slope: Slope of the line
    midpoint : Midpoint is in the form of [x,y], and is also normalized to [0, 1]
    """

    x1, y1, x2, y2 = xy_points[0][0], xy_points[0][1], xy_points[1][
        0], xy_points[1][1]
    slope = (y2 - y1) / (x2 - x1)

    midpoint_x = 0.5
    midpoint_y = slope * (0.5 - x1) + y1
    midpoint = [midpoint_x, midpoint_y]
    return slope, midpoint


def calculate_horizon_error(annotated_horizon, proposed_horizon):
    """
    Calculate the error between the annotated horizon and the proposed horizon

    Args:
    annotated_horizon: list of two points, each point is a list of two elements
    Points are in the form of [x, y], where x and y are normalized to [0, 1]
    proposed_horizon: list of two points, each point is a list of two elements
    Points are in the form of [x, y], where x and y are normalized to [0, 1]

    Returns:
    slope_error: Error in the slope of the lines
    midpoint_error: Error in the midpoint_y of the lines
    """

    slope_annotated, midpoint_annotated = xy_points_to_slope_midpoint(
        annotated_horizon)
    slope_proposed, midpoint_proposed = xy_points_to_slope_midpoint(
        proposed_horizon)

    slope_error = abs(slope_annotated - slope_proposed)
    midpoint_error = abs(midpoint_annotated[1] - midpoint_proposed[1])

    return slope_error, midpoint_error


def calculate_horizon_error_across_sequence(slope_error_list,
                                            midpoint_error_list,
                                            slope_error_jump_threshold,
                                            midpoint_error_jump_threshold):
    """
    Calculate the error statistics across a sequence of frames

    Args:
    slope_error_list: List of errors in the slope of the lines
    midpoint_error_list: List of errors in the midpoint_y of the lines

    Returns:
    average_slope_error: Average error in the slope of the lines
    average_midpoint_error: Average error in the midpoint_y of the lines
    """

    # Calculate the average and standard deviation of the errors
    average_slope_error = np.mean(slope_error_list)
    average_midpoint_error = np.mean(midpoint_error_list)

    stddev_slope_error = np.std(slope_error_list)
    stddev_midpoint_error = np.std(midpoint_error_list)

    # Calculate the maximum errors
    max_slope_error = np.max(slope_error_list)
    max_midpoint_error = np.max(midpoint_error_list)

    # Calculate the differences between errors in successive frames
    diff_slope_error = np.abs(np.diff(slope_error_list))
    diff_midpoint_error = np.abs(np.diff(midpoint_error_list))

    # Calculate the number of jumps in the errors
    num_slope_error_jumps = np.sum(
        diff_slope_error > slope_error_jump_threshold)
    num_midpoint_error_jumps = np.sum(
        diff_midpoint_error > midpoint_error_jump_threshold)

    # Create a dictionary to store the results
    sequence_results = {
        'average_slope_error': average_slope_error,
        'average_midpoint_error': average_midpoint_error,
        'stddev_slope_error': stddev_slope_error,
        'stddev_midpoint_error': stddev_midpoint_error,
        'max_slope_error': max_slope_error,
        'max_midpoint_error': max_midpoint_error,
        'num_slope_error_jumps': num_slope_error_jumps,
        'num_midpoint_error_jumps': num_midpoint_error_jumps
    }

    return sequence_results


def slope_to_roll(slope):
    """
    Convert the slope of the horizon to roll

    Args:
    slope: Slope of the horizon

    Returns:
    roll: Roll in degrees
    """
    roll = np.arctan(slope) * 180 / np.pi
    return roll


def roll_to_slope(roll):
    """
    Convert the roll of the horizon to slope

    Args:
    roll: Roll of the horizon in degrees

    Returns:
    slope: Slope of the horizon
    """
    slope = np.tan(roll * np.pi / 180)
    return slope


def midpoint_to_pitch(midpoint, vertical_fov_degrees):
    """
    Convert the midpoint of the horizon to pitch

    Args:
    midpoint: Midpoint of the horizon
    vertical_fov_degrees: Vertical field of view of the camera in degrees

    Returns:
    pitch: Pitch in degrees
    """
    pitch = midpoint * vertical_fov_degrees
    return pitch


def pitch_to_midpoint(pitch, vertical_fov_degrees):
    """
    Convert the pitch of the horizon to midpoint

    Args:
    pitch: Pitch of the horizon in degrees
    vertical_fov_degrees: Vertical field of view of the camera in degrees

    Returns:
    midpoint: Midpoint of the horizon
    """
    midpoint = pitch / vertical_fov_degrees
    return midpoint
