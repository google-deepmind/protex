# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Computes and prints maximum micro-averaged F1 score.

This metric was used by "ProteInfer, deep networks for protein functional
inference".
"""

from absl import app
from absl import flags
import numpy as np
import sklearn.metrics

from common import jsonl_utils
from eval import eval_utils


FLAGS = flags.FLAGS


flags.DEFINE_string("dataset", "", "Path to jsonl dataset file.")

flags.DEFINE_string("predictions", "", "Path to jsonl predictions file.")


def get_max_f1(true_labels, pred_scores):
  """Return maximum micro-averaged F1 score."""
  true_labels = true_labels.flatten()
  pred_scores = pred_scores.flatten()
  precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(
      true_labels, pred_scores
  )
  # The last values have no associated threshold.
  precisions = precisions[:-1]
  recalls = recalls[:-1]

  # Make denominator robust to zeros.
  denominator = np.where(precisions + recalls == 0, 1, precisions + recalls)
  f1_scores = 2 * precisions * recalls / denominator
  max_f1_score_idx = np.argmax(f1_scores)
  max_threshold = thresholds[max_f1_score_idx]
  max_f1 = f1_scores[max_f1_score_idx]
  print(f"max_threshold: {max_threshold}")
  return max_f1


def main(unused_argv):
  predictions = jsonl_utils.read(FLAGS.predictions)
  dataset = jsonl_utils.read(FLAGS.dataset)
  all_labels = eval_utils.get_all_labels(dataset, predictions)
  true_labels, pred_scores = eval_utils.preprocess_preds(
      dataset, predictions, all_labels
  )
  max_f1 = get_max_f1(true_labels, pred_scores)
  print(f"max_f1: {max_f1}")


if __name__ == "__main__":
  app.run(main)
