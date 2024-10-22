import torch

unit_tests_results = {
    "test_panoptic_pq": {
        "test_compute_metric": {
            "pq_value": torch.tensor(
                [
                    [
                        0.6667,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.9999,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        0.6667,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.9999,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                ],
                device="cuda:0",
                dtype=torch.float64,
            ),
            "rq_value": torch.tensor(
                [
                    [
                        1.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.9999,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        1.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.9999,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                ],
                device="cuda:0",
                dtype=torch.float64,
            ),
            "sq_value": torch.tensor(
                [
                    [
                        0.6667,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        1.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        0.6667,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                    [
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        1.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ],
                ],
                device="cuda:0",
                dtype=torch.float64,
            ),
        }
    }
}