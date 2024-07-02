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

"""Evaluate PFAM predictions.

Metrics used are:
-Family accuracy
-Lifted clan accuracy.
-Average per class family accuracy.

The dataset should be in the jsonl format indicated by convert_pfam.py where
each line is of the form:
 {"accession": "C6WKU9_ACTMD/45-114",
  "sequence": "ARNDCEF...."
  "labels": ["PF03793.19"]}

The predictions should be in jsonl format where each line is of the form:
 {"accession": "C6WKU9_ACTMD/45-114", "predicted_label": "PF03793.19"}

The family to  clan mapping file can be found here:

wget
ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam32.0/Pfam-A.clans.tsv.gz .
gzip -d Pfam-A.clans.tsv.gz
"""

import collections

from absl import app
from absl import flags
import numpy as np
import pandas as pd

from common import jsonl_utils
from eval import eval_pfam_utils

FLAGS = flags.FLAGS


flags.DEFINE_string('dataset', '', 'Path to jsonl dataset file.')

flags.DEFINE_string('predictions', '', 'Path to jsonl predictions file.')

flags.DEFINE_string('clan_mapping', None, 'Path to clan mapping file.')


def convert_predictions_to_df(jsonl_data):
  accessions = []
  predicted_labels = []
  for row in jsonl_data:
    accessions.append(row['accession'])
    predicted_labels.append(row['predicted_label'])

  return pd.DataFrame(
      {'accession': accessions, 'predicted_label': predicted_labels}
  )


def convert_dataset_to_df(jsonl_data):
  accessions = []
  labels = []
  for row in jsonl_data:
    accessions.append(row['accession'])
    assert len(row['labels']) == 1
    labels.append(row['labels'][0])

  return pd.DataFrame({'accession': accessions, 'true_label': labels})


def mean_per_class_accuracy(predictions_dataframe):
  """Compute accuracy of predictions, giving equal weight to all classes.

  Args:
    predictions_dataframe: pandas DataFrame with 3 columns,
      classification_util.PREDICTION_FILE_COLUMN_NAMES.

  Returns:
    float. The average of all class-level accuracies.
  """
  grouped_predictions = collections.defaultdict(list)
  for row in predictions_dataframe.itertuples():
    grouped_predictions[row.true_label].append(row.predicted_label)

  accuracy_per_class = {
      true_label: np.mean(predicted_label == np.array(true_label))
      for true_label, predicted_label in grouped_predictions.items()
  }

  return np.mean(list(accuracy_per_class.values()))


def main(unused_argv):
  predictions = jsonl_utils.read(FLAGS.predictions)
  dataset = jsonl_utils.read(FLAGS.dataset)

  prediction_df = convert_predictions_to_df(predictions)
  reference_df = convert_dataset_to_df(dataset)

  merged_df = prediction_df.merge(
      reference_df, on='accession', how='inner', validate='one_to_one'
  )

  family_accuracy = eval_pfam_utils.raw_unweighted_accuracy(merged_df)
  per_class_family_accuracy = eval_pfam_utils.mean_per_class_accuracy(merged_df)

  print('Family accuracy: %.1f' % (family_accuracy * 100))
  print('Avg. Per-Family accuracy: %.1f' % (per_class_family_accuracy * 100))

  if FLAGS.clan_mapping:
    lifted_clan_accuracy = eval_pfam_utils.get_unweighted_lifted_clan_accuracy(
        merged_df, FLAGS.clan_mapping
    )
    print('Lifted Clan accuracy: %.1f' % (lifted_clan_accuracy * 100))


if __name__ == '__main__':
  app.run(main)
