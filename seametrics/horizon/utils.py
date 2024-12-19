from typing import List

import fiftyone as fo
import numpy as np

from sklearn.linear_model import LinearRegression


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

    if all_slope_errors_none:
        average_slope_error = None
        stddev_slope_error = None
        max_slope_error = None
        num_slope_error_jumps = None
        average_slope_error_deg = None
        stddev_slope_error_deg = None
        max_slope_error_deg = None
    else:
        average_slope_error = np.mean(filtered_slope_error_list)
        stddev_slope_error = np.std(filtered_slope_error_list)
        max_slope_error = np.max(filtered_slope_error_list)

        slope_hist = np.histogram(
            filtered_slope_error_list,
            bins=int(180/10),
            range=(0, 180)
        )

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
    else:
        average_midpoint_error = np.mean(filtered_midpoint_error_list)
        stddev_midpoint_error = np.std(filtered_midpoint_error_list)
        max_midpoint_error = np.max(filtered_midpoint_error_list)

        midpoint_hist = np.histogram(
            filtered_midpoint_error_list,
            bins=int(512/20),
            range=(0, 512)
        )

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
    if vertical_fov_degrees is None:
        return None
    midpoint = pitch / vertical_fov_degrees
    return midpoint

def get_horizon_from_water(mask: np.ndarray) -> List[List[float]]:
    """
    Generate horizon from water mask by fitting a linear regression
    to all x-values and their respective top-most non-zero y-value.

    Args:
        mask (np.ndarray): binary mask of water in shape of image

    Returns:
        List[List[float]]: horizon defined by [[0, y_1], [1, y_2]],
            where y_1, y_2 € [0,1].
    """
    row, col = np.nonzero(mask)
    row = row.astype(np.float32)/ mask.shape[0]
    col = col.astype(np.float32)/ mask.shape[1]
    xs = np.unique(col)
    ys = np.array([min(row[col==x]) for x in xs])
    reg = LinearRegression().fit(xs[..., np.newaxis], ys)
    min_y, max_y = reg.predict([[0],[1]])
    return [[0, min_y], [1, max_y]]

def horizon_for_sequence(seq: fo.DatasetView, field: str) -> List[List[List[float]]]:
    """
    Extract horizons for all frames of sequence.

    Args:
        seq (fo.DatasetView): FiftyOne view holding all frames of sequence.
        field (str): Field to extract annotations from.

    Returns:
        List[List[List[float]]]: list holding all horizos in shape where the
            length is the number of frames in the sequence and horizons[i] is
            the parameterization of the horizon in frame i parametrized by
            two points in format [[0, y_1], [1, y_2]].
    """
    horizons = []
    for sample in seq:
        if hasattr(sample[field], "polylines") and (sample[field].polylines is not None and len(sample[field].polylines) > 0):
            horizon = sample[field].polylines[0].points[0]
        elif hasattr(sample[field], "detections"):
            horizon = None #horizon is initialized at None in case no water mask is present
            h, w = sample.metadata.height, sample.metadata.width
            for det in sample[field].detections:
                if det.label == "WATER":
                    if (not hasattr(det, "mask")) or (det.mask is None):
                        raise ValueError("Non-segmentation dataset.")
                    mask_water = det["mask"]
                    full_mask = np.zeros((h, w))
                    x = int(det["bounding_box"][0] * w)
                    y = int(det["bounding_box"][1] * h)
                    w_b = int(det["bounding_box"][2] * w)
                    h_b = int(det["bounding_box"][3] * h)
                    full_mask[y:(y+h_b), x:(x+w_b):] = np.array(mask_water)
                    horizon = get_horizon_from_water(full_mask)
        else:
            horizon = None # if no polyline or water mask is present, horizon is None
        horizons.append(horizon)

    return horizons

