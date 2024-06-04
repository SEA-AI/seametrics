## Horizon Metrics

This metric uses `seametrics.horizon.HorizonMetrics` to calculate the slope and midpoint errors.

## How to Use

To utilize horizon-metrics effectively, start by installing the necessary dependencies using the provided pip command. Once installed, import the evaluate library into your Python environment. Then, use this metric to evaluate your horizon prediction models. Ensure that both ground truth and prediction points are correctly formatted before computing the result. Finally, analyze the computed result to gain insights into the performance of your prediction models.

### Getting Started

To get started with horizon-metrics, make sure you have the necessary dependencies installed. This metric relies on the `seametrics` libraries.

### Installation

```sh
  pip install evaluate git+https://github.com/SEA-AI/seametrics@develop
```

### Basic Usage

This is how you can quickly evaluate your horizon prediction models using the horizon-metrics:

> [!IMPORTANT]  
> The vertical_fov_degrees and heigt parameters are required. The default value for roll_rehsold is 0.5 and for pitch_threshold it is 0.1.

> [!IMPORTANT]  
> The horizon metric should be calculated per sequence. Make sure that the vertical_fov and height are consistent across the inputs and do not change.

##### Use artifical data for testing

```python
ground_truth_points = [[[0.0, 0.5384765625], [1.0, 0.4931640625]],
                       [[0.0, 0.53796875], [1.0, 0.4928515625]],
                       [[0.0, 0.5374609375], [1.0, 0.4925390625]],
                       [[0.0, 0.536953125], [1.0, 0.4922265625]],
                       [[0.0, 0.5364453125], [1.0, 0.4919140625]]]

prediction_points = [[[0.0, 0.5428930956049597], [1.0, 0.4642497615378973]],
                     [[0.0, 0.5428930956049597], [1.0, 0.4642497615378973]],
                     [[0.0, 0.523573113510805], [1.0, 0.47642688648919496]],
                     [[0.0, 0.5200016849393765], [1.0, 0.4728554579177664]],
                     [[0.0, 0.523573113510805], [1.0, 0.47642688648919496]]]
```

##### Load data from fiftyone

```python
# Load data from fiftyone
sequence = "Sentry_2022_11_PROACT_CELADON_7.5M_MOB_2022_11_25_12_29_48"
dataset_name = "SENTRY_VIDEOS_DATASET_QA"
sequence_view = fo.load_dataset(dataset_name).match(F("sequence") == sequence)
sequence_view = sequence_view.select_group_slices("thermal_wide")

# Get the ground truth points
polylines_gt = sequence_view.values("frames.ground_truth_pl")
ground_truth_points = [
    line["polylines"][0]["points"][0] if line is not None else None
    for line in polylines_gt[0]
]

# Get the predicted points
polylines_pred = sequence_view.values(
    "frames.ahoy-IR-b2-whales__XAVIER-AGX-JP46_pl_TnFoV")
prediction_points = [
    line["polylines"][0]["points"][0] if line is not None else None
    for line in polylines_pred[0]
```

##### Calculate horizon metrics

```python
from seametrics.horizon.horizon import HorizonMetrics

# Create the metrics object
metrics = HorizonMetrics(vertical_fov_degrees=25.6, height=512, roll_threshold=0.5, pitch_threshold=0.1)

# Set ground truth and predictions
metrics.update(predictions=prediction_points,
               ground_truth_det=ground_truth_points)

# Compute metrics
metrics.compute()

```

This is output the evalutaion metrics for your horizon prediciton model:

```console
{'average_slope_error': 0.39394822776758726,
 'average_midpoint_error': 0.0935801366906932,
 'average_midpoint_error_px': 1.871602733813864,
 'stddev_slope_error': 0.3809031270343266,
 'stddev_midpoint_error': 0.23003871087476538,
 'stddev_midpoint_error_px': 4.6007742174953075,
 'max_slope_error': 3.5549008029526132,
 'max_midpoint_error': 2.515424321301225,
 'max_midpoint_error_px': 50.3084864260245,
 'num_slope_error_jumps': 173,
 'num_midpoint_error_jumps': 205,
 'detection_rate': 0.2606486908948808}
```

### Output Values

The metric includes the following performance metrics for horizon prediction:

- **average_slope_error**: Measures the average difference in slope between the predicted and ground truth horizon in degree.
- **average_midpoint_error**: Represents the average difference in midpoint position between the predicted and ground truth horizon.
- **average_midpoint_error_px**: Represents the average difference in midpoint position between the predicted and ground truth horizon, measured in pixels.
- **stddev_slope_error**: Indicates the variability of errors in slope between the predicted and ground truth horizon in degree.
- **stddev_midpoint_error**: Quantifies the variability of errors in midpoint position between the predicted and ground truth horizon in degree.
- **stddev_midpoint_error_px**: Quantifies the variability of errors in midpoint position between the predicted and ground truth horizon, measured in pixels.
- **max_slope_error**: Represents the maximum difference in slope between the predicted and ground truth horizon in degree.
- **max_midpoint_error**: Indicates the maximum difference in midpoint position between the predicted and ground truth horizon in degree.
- **max_midpoint_error_px**: Indicates the maximum difference in midpoint position between the predicted and ground truth horizon, measured in pixels.
- **num_slope_error_jumps**: Calculates the differences between errors in successive frames for the slope. It then counts the number of jumps in these errors by comparing the absolute differences to a specified threshold.
- **num_midpoint_error_jumps**: Calculates the differences between errors in successive frames for the midpoint. It then counts the number of jumps in these errors by comparing the absolute differences to a specified threshold.
- **detection_rate**: Measures the proportion of frames in which the horizon is successfully detected out of the total number of frames.
