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

"""Converts ProteInfer file to common jsonl format.

This dataset was proposed in the paper:
"ProteInfer, deep networks for protein functional inference":
https://google-research.github.io/proteinfer/

The dataset files are available here:
https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/proteinfer/datasets/.
"""

import random

from absl import app
from absl import flags
import tensorflow as tf

from common import jsonl_utils


FLAGS = flags.FLAGS


flags.DEFINE_string("input", "", "Path to ProteInfer file.")

flags.DEFINE_string("output", "", "Output location for jsonl file.")

flags.DEFINE_enum("labels", "ec", ["ec", "go"], "Which labels to use.")

flags.DEFINE_float("sample_frac", 1.0, "Sample a fraction of the input.")


def load_tf_examples(path):
  filepaths = tf.io.gfile.glob(path)
  dataset = tf.data.TFRecordDataset(filepaths)
  records = []
  for raw_record in dataset:
    record = tf.train.Example.FromString(raw_record.numpy())
    records.append(record)
  return records


def get_bytes_feature(example: tf.train.Example, key: str) -> bytes:
  return example.features.feature[key].bytes_list.value[0]


def get_text_feature(example: tf.train.Example, key: str) -> str:
  return get_bytes_feature(example, key).decode("utf-8")


def get_repeated_text_feature(example: tf.train.Example, key: str) -> list[str]:
  values = []
  for value in example.features.feature[key].bytes_list.value:
    values.append(value.decode("utf-8"))
  return values


def filter_labels(labels):
  if FLAGS.labels == "go":
    return [label for label in labels if label.startswith("GO:")]
  elif FLAGS.labels == "ec":
    return [label for label in labels if label.startswith("EC:")]
  else:
    raise ValueError("Unknown label type: %s" % FLAGS.labels)


def load_examples(path):
  """Load tfrecord file."""
  examples = []
  tf_examples = load_tf_examples(path)
  for example in tf_examples:
    sequence = get_text_feature(example, "sequence")
    accession = get_text_feature(example, "id")
    labels = get_repeated_text_feature(example, "label")
    labels = filter_labels(labels)
    example = {
        "sequence": sequence,
        "accession": accession,
        "labels": labels,
    }
    examples.append(example)
  return examples


def main(unused_argv):
  examples = []
  for file_path in tf.io.gfile.glob(FLAGS.input):
    examples.extend(load_examples(file_path))
  if FLAGS.sample_frac < 1.0:
    cutoff = int(FLAGS.sample_frac * len(examples))
    random.shuffle(examples)
    examples = examples[:cutoff]
  jsonl_utils.write(FLAGS.output, examples)


if __name__ == "__main__":
  app.run(main)
