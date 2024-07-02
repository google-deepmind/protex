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

r"""Converts the PFAM seed clustered split.

This data split is from the paper:
"Using deep learning to annotate the protein universe"
https://www.nature.com/articles/s41587-021-01179-w

The data can be obtained with the following command:

wget
https://storage.googleapis.com/brain-genomics-public/research/proteins/pfam/clustered_split/pfam_clustered_split__train_dev_test.tar.gz
"""

from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf

from common import jsonl_utils


flags.DEFINE_string("input", "", "Path to PFAM input file pattern.")

flags.DEFINE_string("output", "", "Output path for json file.")

FLAGS = flags.FLAGS


def read_df(file_pattern):
  file_paths = tf.io.gfile.glob(file_pattern)
  dfs = []
  for file_path in file_paths:
    with tf.io.gfile.GFile(file_path, "r") as f:
      df = pd.read_csv(f)
      dfs.append(df)

  df = pd.concat(dfs).reset_index(drop=True)
  return df


def convert_to_example(accession, sequence, label):
  return {
      "accession": accession,
      "sequence": sequence,
      "labels": [label],
  }


def main(unused_argv):
  data_df = read_df(FLAGS.input)
  print(f"Loaded {data_df.shape[0]} rows.")

  accessions = data_df["sequence_name"].values.tolist()
  sequences = data_df["sequence"].values.tolist()
  labels = data_df["family_accession"].values.tolist()

  examples = [
      convert_to_example(accession, sequence, label)
      for accession, sequence, label in zip(accessions, sequences, labels)
  ]
  jsonl_utils.write(FLAGS.output, examples)


if __name__ == "__main__":
  app.run(main)
