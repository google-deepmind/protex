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

"""Converts CLEAN files to common format.

This dataset was proposed in the paper:
"Enzyme Function Prediction using Contrastive Learning"
https://www.science.org/doi/10.1126/science.adf2465

The dataset files are available here:
https://github.com/tttianhao/CLEAN/tree/main/app/data
"""

import csv

from absl import app
from absl import flags
import tensorflow as tf

from protex.common import jsonl_utils


FLAGS = flags.FLAGS


flags.DEFINE_string("input", "", "Path to input file.")

flags.DEFINE_string("output", "", "Output location for jsonl file.")


def load_tsv(filepath):
  dicts = []
  with tf.io.gfile.GFile(filepath, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row_dict in reader:
      dicts.append(row_dict)
  return dicts


def convert_to_example(row_dict):
  return {
      "accession": row_dict["Entry"],
      "sequence": row_dict["Sequence"],
      "labels": row_dict["EC number"].split(";"),
  }


def main(unused_argv):
  rows = load_tsv(FLAGS.input)
  print(f"Loaded {len(rows)} rows.")
  examples = [convert_to_example(row_dict) for row_dict in rows]
  jsonl_utils.write(FLAGS.output, examples)


if __name__ == "__main__":
  app.run(main)
