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

"""Utilities for evaluating pfam predictions."""

import collections

import numpy as np
import pandas as pd
import tensorflow as tf


def read_pfam_clan_file(path: str) -> pd.DataFrame:
  """Parses pfam clan tsv file.

  Args:
    path: Path to tsv clan file.

  Returns:
    pd.DataFrame with columns family_accession (str), clan_accession (str),
    clan_description (str), family_name (str), family_description (str).
  """
  with tf.io.gfile.GFile(path, 'r') as f:
    return pd.read_csv(
        f,
        names=[
            'family_accession',
            'clan_accession',
            'clan_description',
            'family_name',
            'family_description',
        ],
        sep='\t',
        # Some fields are missing, and we want to keep those
        # as empty strings instead of the default behavior,
        # which is to convert them to NaNs.
        keep_default_na=False,
    )


def family_to_clan_mapping(path: str) -> dict[str, str]:
  """Parse tsv contents, returning dict from pfam family to clan accession.

  Families without a clan will get their own clan in
      the returned dictionary, with clan name == to the accession (e.g. PF12345
      -> PF12345).

  Args:
    path: Path to tsv clan file.

  Returns:
    dict from string to string, e.g. {'PF12345': 'CL9999'}.
  """
  dataframe = read_pfam_clan_file(path)

  dataframe['clan_accession'] = dataframe.apply(
      axis='columns',
      func=lambda row: row.clan_accession  # pylint: disable=g-long-lambda
      if row.clan_accession
      else row.family_accession,
  )

  # Filter family names without clans (they are are stored in the csv
  # as empty strings). If we're using lifted clan semantics, every family will
  # have a clan (see docstring).
  return dict(
      (family_id, clan_id)  # pylint: disable=g-complex-comprehension
      for family_id, clan_id in zip(
          dataframe['family_accession'].values,
          dataframe['clan_accession'].values,
      )
      if clan_id
  )


def raw_unweighted_accuracy(
    predictions_df: pd.DataFrame,
    true_label_column: str = 'true_label',
    predicted_label_column: str = 'predicted_label',
) -> float:
  """Compute accuracy, regardless of which class each prediction corresponds to.

  Args:
    predictions_df: pandas DataFrame with at least 2 columns, true_label and
      predicted_label.
    true_label_column: Column name of true labels.
    predicted_label_column: str. Column name of predicted labels.

  Returns:
    Accuracy.
  """
  num_correct = (
      predictions_df[true_label_column]
      == predictions_df[predicted_label_column]
  ).sum()
  total = len(predictions_df)
  return num_correct / total


def mean_per_class_accuracy(
    predictions_df: pd.DataFrame,
    true_label_column: str = 'true_label',
    predicted_label_column: str = 'predicted_label',
) -> float:
  """Compute accuracy of predictions, giving equal weight to all classes.

  Args:
    predictions_df: pandas DataFrame with at least 2 columns, true_label and
      predicted_label.
    true_label_column: Column name of true labels.
    predicted_label_column: str. Column name of predicted labels.

  Returns:
    The average of all class-level accuracies.
  """
  grouped_predictions = collections.defaultdict(list)
  for _, row in predictions_df.iterrows():
    grouped_predictions[row[true_label_column]].append(
        row[predicted_label_column]
    )

  accuracy_per_class = {
      true_label: np.mean(predicted_label == np.array(true_label))
      for true_label, predicted_label in grouped_predictions.items()
  }

  return np.mean(list(accuracy_per_class.values()))


def get_unweighted_lifted_clan_accuracy(
    predictions_df: pd.DataFrame,
    clan_mapping_path: str,
    true_label_column: str = 'true_label',
    predicted_label_column: str = 'predicted_label',
):
  """Compute accuracy, where each label is mapped to its clan.

  Args:
    predictions_df: pandas DataFrame with at least 2 columns, true_label and
      predicted_label.
    clan_mapping_path: Path to tsv clan file.
    true_label_column: Column name of true labels.
    predicted_label_column: Column name of predicted labels.

  Returns:
    Lifted Clan Accuracy.
  """

  def pfam_accession_helper(x):
    x_split = x.split('.')
    assert len(x_split) == 2
    return x_split[0]

  family_to_clan = family_to_clan_mapping(clan_mapping_path)

  predictions_df['true_clan_label'] = predictions_df[true_label_column].apply(
      lambda x: family_to_clan.get(
          pfam_accession_helper(x), pfam_accession_helper(x)
      )
  )

  predictions_df['predicted_clan_label'] = predictions_df[
      predicted_label_column
  ].apply(
      lambda x: family_to_clan.get(
          pfam_accession_helper(x), pfam_accession_helper(x)
      )
  )

  return raw_unweighted_accuracy(
      predictions_df,
      true_label_column='true_clan_label',
      predicted_label_column='predicted_clan_label',
  )
