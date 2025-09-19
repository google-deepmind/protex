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

"""Common utilities for evaluation."""

import collections

import numpy as np


NEG_INF = -1e9


def preprocess_preds(dataset, scores, all_labels):
  """Return predictions and ground truth labels in common format."""
  # Map of accession to map of label to score.
  accession_to_predictions = collections.defaultdict(dict)
  for row in scores:
    score = float(row["score"])
    accession = row["inputs"]["accession"]
    label = row["inputs"]["label"]
    accession_to_predictions[accession][label] = score

  true_labels = []
  pred_scores = []
  for row in dataset:
    accession = row["accession"]
    predictions = accession_to_predictions[accession]
    gold_labels = set(row["labels"])
    true_labels_row = []
    pred_scores_row = []
    for label in all_labels:
      true_label = 1 if label in gold_labels else 0
      pred_score = predictions.get(label, NEG_INF)
      true_labels_row.append(true_label)
      pred_scores_row.append(pred_score)
    true_labels.append(true_labels_row)
    pred_scores.append(pred_scores_row)

  return np.array(true_labels), np.array(pred_scores)


def get_all_labels(dataset, predictions):
  """Return union of labels in predictions and dataset."""
  all_labels = set()
  for row in dataset:
    for label in row["labels"]:
      all_labels.add(label)
  for row in predictions:
    all_labels.add(row["inputs"]["label"])
  return all_labels
