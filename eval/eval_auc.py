# Copyright 2024 DeepMind Technologies Limited
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

"""Computes and prints weighted AUC metric.

This metric was used in "Enzyme Function Prediction using Contrastive Learning".
"""

from absl import app
from absl import flags
import sklearn.metrics

from protex.common import jsonl_utils
from protex.eval import eval_utils


FLAGS = flags.FLAGS


flags.DEFINE_string("dataset", "", "Path to jsonl dataset file.")

flags.DEFINE_string("predictions", "", "Path to jsonl predictions file.")


def get_test_labels(dataset):
  all_labels = set()
  for row in dataset:
    for label in row["labels"]:
      all_labels.add(label)
  return all_labels


def get_auc(true_labels, pred_scores):
  """Return AUC."""
  return sklearn.metrics.roc_auc_score(
      true_labels, pred_scores, average="weighted"
  )


def main(unused_argv):
  predictions = jsonl_utils.read(FLAGS.predictions)
  dataset = jsonl_utils.read(FLAGS.dataset)
  # Only labels occurring in the test set are considered for this metric.
  all_labels = get_test_labels(dataset)
  true_labels, pred_scores = eval_utils.preprocess_preds(
      dataset, predictions, all_labels
  )
  auc = get_auc(true_labels, pred_scores)
  print(f"auc: {auc}")


if __name__ == "__main__":
  app.run(main)
