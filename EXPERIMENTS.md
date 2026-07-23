# Experiment report template

Record each run with the exact configuration and checkpoint committed or
archived alongside the result table.

| Run | Split | Input | Epochs | Best checkpoint rule | F1 | Precision | Recall | mIoU | Notes |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| RAVEN-sequence | sequence | raw ADC, no DC removal | 150 | validation loss |  |  |  |  |  |
| RAVEN-random | random 80/20 | raw ADC, no DC removal | 80 | validation loss |  |  |  |  |  |

For a detection threshold sweep, report the confidence threshold and IoU
threshold separately. Confidence filters predicted objects; IoU determines
whether a retained prediction matches a ground-truth object.

Suggested artifacts for every reported run:

- `config.json`
- `split.json`
- `history.csv`
- the selected checkpoint filename and epoch
- `evaluation_train/metrics.json`
- `evaluation_val/metrics.json`
- `evaluation_test/metrics.json` when a distinct test set exists
- each evaluation directory's `detection_sweep.csv`

