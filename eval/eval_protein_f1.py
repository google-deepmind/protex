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

"""Computes and prints maximum protein-centric F1 score.

This is a commonly used metric for evaluating protein function prediction
methods. Details can be found in "A large-scale evaluation of computational
protein function prediction" (https://www.nature.com/articles/nmeth.2340).

Note that more efficient implementations exist, such as this one that depends
on PyTorch:
https://github.com/DeepGraphLearning/torchdrug/blob/6066fbd82360abb5f270cba1eca560af01b8cc90/torchdrug/metrics/metric.py#L234

However, our goal was to implement the metric in a way that is easier to
understand and verify and without additional dependencies.
"""

from absl import app
from absl import flags
import numpy as np

from common import jsonl_utils
from eval import eval_utils


FLAGS = flags.FLAGS


flags.DEFINE_string("dataset", "", "Path to jsonl dataset file.")

flags.DEFINE_string("predictions", "", "Path to jsonl predictions file.")

flags.DEFINE_integer(
    "precision",
    2,
    "Round scores to this many decimals if >0."
    "Helps speed up computation by considering fewer thresholds.",
)


def get_counts(y_true, y_pred):
  y_true_and_pred = y_true * y_pred
  tp = np.sum(y_true_and_pred, axis=1)
  fp = np.sum(y_pred, axis=1) - tp
  fn = np.sum(y_true, axis=1) - tp
  return tp, fp, fn


def get_protein_centric_f1(y_true, y_pred):
  """Computes protein-centric F1 score."""
  tp, fp, fn = get_counts(y_true, y_pred)
  # If there are no predictions, then precision is undefined and does not count
  # towards the overall average, per the definition of protein-centric F1.
  # Set undefined values to 0.
  precision_num = np.divide(
      tp, tp + fp, out=np.zeros_like(tp, dtype=np.float32), where=tp != 0
  )
  precision_denom = (tp + fp) > 0
  recall = tp / (tp + fn)
  precision_avg = np.sum(precision_num) / np.sum(precision_denom)
  recall_avg = np.mean(recall)
  f1 = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)
  return f1


def get_thresholds(preds):
  return np.sort(np.unique(preds.flatten()))


def get_max_protein_centric_f1(pred_scores, target_labels):
  pred_scores = np.array(pred_scores)
  target_labels = np.array(target_labels)
  thresholds = get_thresholds(pred_scores)
  print(f"num thresholds: {len(thresholds)}")
  f1_scores = []
  for threshold in thresholds:
    pred_labels = pred_scores >= threshold
    f1_scores.append(get_protein_centric_f1(target_labels, pred_labels))
  return max(f1_scores)


def main(unused_argv):
  predictions = jsonl_utils.read(FLAGS.predictions)
  dataset = jsonl_utils.read(FLAGS.dataset)
  all_labels = eval_utils.get_all_labels(dataset, predictions)
  true_labels, pred_scores = eval_utils.preprocess_preds(
      dataset, predictions, all_labels
  )
  if FLAGS.precision > 0:
    pred_scores = np.around(pred_scores, decimals=FLAGS.precision)

  max_f1 = get_max_protein_centric_f1(pred_scores, true_labels)
  print(f"max_f1: {max_f1}")


if __name__ == "__main__":
  app.run(main)
