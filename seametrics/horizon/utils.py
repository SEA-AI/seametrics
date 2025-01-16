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
                                            midpoint_error_jump_threshold,
                                            vertical_fov_degrees, height):
    """
    Calculate the horizon error across a sequence of frames.

    Parameters
    ----------
    slope_error_list : list
        List of slope errors for each frame.
    midpoint_error_list : list
        List of midpoint errors for each frame.
    slope_error_jump_threshold : float
        Threshold for detecting jumps in slope errors.
    midpoint_error_jump_threshold : float
        Threshold for detecting jumps in midpoint errors.
    vertical_fov_degrees : float or None
        Vertical field of view in degrees.
        None if only the pixel metrics are of interest.
    height : int
        Height of the image.

    Returns
    -------
    dict
        A dictionary containing the following results:
        - 'average_slope_error': The average slope error in degrees.
        - 'average_midpoint_error': The average midpoint error in degrees.
        - 'average_midpoint_error_px': The average midpoint error in pixels.
        - 'stddev_slope_error': The standard deviation of slope errors in degrees.
        - 'stddev_midpoint_error': The standard deviation of midpoint errors in degrees.
        - 'stddev_midpoint_error_px': The standard deviation of midpoint errors in pixels.
        - 'max_slope_error': The maximum slope error in degrees.
        - 'max_midpoint_error': The maximum midpoint error in degrees.
        - 'max_midpoint_error_px': The maximum midpoint error in pixels.
        - 'num_slope_error_jumps': The number of jumps in slope errors.
        - 'num_midpoint_error_jumps': The number of jumps in midpoint errors.

        If vertical_fov_degrees is None, all 
    """
    #check if slope_error_list and midpoint_error_list are full of Nones
    all_slope_errors_none = all(x is None for x in slope_error_list)
    all_midpoint_errors_none = all(x is None for x in midpoint_error_list)

    filtered_slope_error_list = [x for x in slope_error_list if x is not None]
    filtered_midpoint_error_list = [x for x in midpoint_error_list if x is not None]

    predicted_samples = min(len(filtered_midpoint_error_list), len(filtered_slope_error_list))
    mp_bins = np.concatenate((np.arange(start=0,stop=10,step=1), [10, 15, 20, 50, 100, 250, 400, 640]))
    roll_bins = np.concatenate((np.arange(start=0,stop=3,step=0.1), np.arange(start=3,stop=5,step=0.2), [5, 10, 20, 100, 180]))

    if all_slope_errors_none:
        average_slope_error = None
        stddev_slope_error = None
        max_slope_error = None
        num_slope_error_jumps = None
        average_slope_error_deg = None
        stddev_slope_error_deg = None
        max_slope_error_deg = None
        slope_hist = (None, None)
    else:
        average_slope_error = np.mean(filtered_slope_error_list)
        stddev_slope_error = np.std(filtered_slope_error_list)
        max_slope_error = np.max(filtered_slope_error_list)

        roll = [slope_to_roll(slope_err) for slope_err in filtered_slope_error_list]

        slope_hist = np.histogram(roll, bins=roll_bins)

        # Calculate the differences between errors in successive frames
        diff_slope_error = np.abs(np.diff(filtered_slope_error_list))
        # Calculate the number of jumps in the errors
        if slope_error_jump_threshold:
            num_slope_error_jumps = np.sum(diff_slope_error > slope_error_jump_threshold)
        else:
            num_slope_error_jumps = None
        
        average_slope_error_deg = slope_to_roll(average_slope_error)
        stddev_slope_error_deg = slope_to_roll(stddev_slope_error)
        max_slope_error_deg = slope_to_roll(max_slope_error)


    if all_midpoint_errors_none:
        average_midpoint_error = None
        stddev_midpoint_error = None
        max_midpoint_error = None
        num_midpoint_error_jumps = None
        average_midpoint_error_deg = None
        stddev_midpoint_error_deg = None
        max_midpoint_error_deg = None
        average_midpoint_error_px = None
        stddev_midpoint_error_px = None
        max_midpoint_error_px = None
        midpoint_hist = (None, None)
    else:
        average_midpoint_error = np.mean(filtered_midpoint_error_list)
        stddev_midpoint_error = np.std(filtered_midpoint_error_list)
        max_midpoint_error = np.max(filtered_midpoint_error_list)

        mp_abs = [mp*height for mp in filtered_midpoint_error_list]
        midpoint_hist = np.histogram(mp_abs, bins=mp_bins)

        # Calculate the differences between errors in successive frames
        diff_midpoint_error = np.abs(np.diff(filtered_midpoint_error_list))
        # Calculate the number of jumps in the errors
        if midpoint_error_jump_threshold:
            num_midpoint_error_jumps = np.sum(diff_midpoint_error > midpoint_error_jump_threshold)
        else:
            num_midpoint_error_jumps = None
        
        # Tranform metrics
        average_midpoint_error_px = average_midpoint_error * height
        stddev_midpoint_error_px = stddev_midpoint_error * height
        max_midpoint_error_px = max_midpoint_error * height

        if vertical_fov_degrees is not None:
            average_midpoint_error_deg = midpoint_to_pitch(average_midpoint_error,
                                                        vertical_fov_degrees)
            stddev_midpoint_error_deg = midpoint_to_pitch(stddev_midpoint_error,
                                                        vertical_fov_degrees)
            max_midpoint_error_deg = midpoint_to_pitch(max_midpoint_error,
                                                    vertical_fov_degrees)
        else:
            average_midpoint_error_deg = None
            stddev_midpoint_error_deg = None
            max_midpoint_error_deg = None
    
    if not (all_midpoint_errors_none or all_slope_errors_none):
        tp_over_thresholds = np.array(
            [
                [
                    ((mp_abs <= mpth) & (roll <= rth)).sum() for mpth in mp_bins
                ] for rth in roll_bins
            ])
    else:
        tp_over_thresholds = np.zeros((len(roll_bins), len(mp_bins)))

    # Create a dictionary to store the results
    sequence_results = {
        'average_slope_error': average_slope_error_deg,
        'average_midpoint_error': average_midpoint_error_deg,
        'average_midpoint_error_px': average_midpoint_error_px,
        'midpoint_hist': midpoint_hist,
        'slope_hist': slope_hist,
        'stddev_slope_error': stddev_slope_error_deg,
        'stddev_midpoint_error': stddev_midpoint_error_deg,
        'stddev_midpoint_error_px': stddev_midpoint_error_px,
        'max_slope_error': max_slope_error_deg,
        'max_midpoint_error': max_midpoint_error_deg,
        'max_midpoint_error_px': max_midpoint_error_px,
        'num_slope_error_jumps': num_slope_error_jumps,
        'num_midpoint_error_jumps': num_midpoint_error_jumps,
        'tp_over_thresholds': tp_over_thresholds,
        'predicted_samples': predicted_samples,
        'samples': len(slope_error_list)
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
    if vertical_fov_degrees is None:
        return None
    midpoint = pitch / vertical_fov_degrees
    return midpoint

