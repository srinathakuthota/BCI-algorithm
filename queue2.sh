#!/bin/bash
# Experiment queue (resumable: rows already in results/*.csv are skipped).
set -x
R="python3 run_experiments.py"
CORE="dual conv1d_only fbcsp_mlp fbcsp_svm fbcsp_knn fbcsp_lda"
for D in 2a 2b; do
  $R --protocol subject --dataset $D --models all --seeds 0
  $R --protocol subject --dataset $D --models $CORE --seeds 1 2
done
for D in 2a 2b; do
  $R --protocol loso --dataset $D --models $CORE eegnet --seeds 0
done
for D in 2a 2b; do
  $R --protocol pooled --dataset $D --models all --seeds 0 --noise
  $R --protocol pooled --dataset $D --models $CORE --seeds 1 2 3 4 --noise
done
$R --protocol loso --dataset 2a --models dual --seeds 1 2
$R --protocol loso --dataset 2b --models dual --seeds 1 2
$R --protocol loso --dataset 2a --models cnn_shallow lstm --seeds 0
$R --protocol loso --dataset 2b --models cnn_shallow lstm --seeds 0
echo QUEUE_DONE
