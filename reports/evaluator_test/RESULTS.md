# Evaluator results: evaluator_test

10 trips; 19,674 cleaned GPS observations. The committed model and HMM configuration were used without training or tuning. Reference routes were loaded only after prediction for scoring.

Reproduce from the repository root:

```sh
python scripts/evaluate_dataset.py --data data/evaluator_test
```

Accuracy below is observation-weighted; route metrics are means across trips, matching the student-test summary protocol.

| method           |   pointwise_accuracy_obs_weighted |   pointwise_accuracy_trip_mean |   route_continuity |   route_precision |   route_recall |   route_f1 |   trips_with_acc_ge_95 |
|:-----------------|----------------------------------:|-------------------------------:|-------------------:|------------------:|---------------:|-----------:|-----------------------:|
| Classifier only  |                            0.9125 |                         0.9357 |             0.9340 |            0.8969 |         0.8775 |     0.8771 |                      5 |
| Classifier + HMM |                            0.9589 |                         0.9575 |             0.9643 |            0.9492 |         0.8789 |     0.9091 |                      7 |

| trip_id   | n_obs   | clf_accuracy   | hmm_accuracy   | hmm_route_continuity   | hmm_route_f1   | hmm_breaks   | time_total_s   |
|:----------|:--------|:---------------|:---------------|:-----------------------|:---------------|:-------------|:---------------|
| 00000015  | 6854    | 0.831          | 0.9488         | 0.998                  | 0.8867         | 0            | 1.6402         |
| 00000017  | 1346    | 0.9391         | 0.9309         | 0.8699                 | 0.8486         | 0            | 1.5953         |
| 00000023  | 458     | 0.9432         | 0.9585         | 0.9869                 | 0.8814         | 0            | 0.2652         |
| 00000030  | 654     | 0.8349         | 0.8869         | 0.9142                 | 0.8094         | 0            | 0.5481         |
| 00000049  | 2642    | 0.9928         | 0.9962         | 0.9943                 | 0.9801         | 0            | 1.0909         |
| 00000059  | 1031    | 0.9641         | 0.9661         | 0.9301                 | 0.8614         | 0            | 0.7563         |
| 00000067  | 916     | 0.964          | 0.9651         | 0.9913                 | 0.9419         | 0            | 0.2653         |
| 00000073  | 1144    | 0.986          | 0.9869         | 0.9913                 | 0.9764         | 0            | 0.6606         |
| 00000087  | 1399    | 0.96           | 0.9843         | 0.9886                 | 0.9621         | 0            | 1.563          |
| 00000094  | 3230    | 0.9421         | 0.9514         | 0.9789                 | 0.943          | 0            | 1.1952         |

Timings cover candidate generation, classifier inference and HMM decoding on this laptop; they exclude file loading and reference scoring. Raw trip data is excluded from Git. Artifact/input hashes and runtime versions are recorded in `provenance.json`.
