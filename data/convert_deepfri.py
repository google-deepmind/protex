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

"""Converts PDB-based EC data.

This data is from the paper:
"Structure-based protein function prediction using graph convolutional networks"
https://www.nature.com/articles/s41467-021-23303-9

The data is available here:
https://github.com/flatironinstitute/DeepFRI/tree/master/preprocessing/data
"""

from absl import app
from absl import flags
import tensorflow as tf

from protex.common import jsonl_utils


FLAGS = flags.FLAGS


flags.DEFINE_string("split", "", "Path to split file.")

flags.DEFINE_string("sequences", "", "Path to sequences file.")

flags.DEFINE_string("annotations", "", "Path to annotations file.")

flags.DEFINE_string("output", "", "Path to write jsonl.")


def read_txt(filepath):
  """Read newline separated text file."""
  rows = []
  with tf.io.gfile.GFile(filepath, "r") as tsv_file:
    for line in tsv_file:
      line = line.rstrip("\n")
      rows.append(line)
  print("Loaded %s rows from %s." % (len(rows), filepath))
  return rows


def read_fasta(filepath):
  """Parse FASTA file."""
  description = None
  sequence = ""
  with tf.io.gfile.GFile(filepath, "r") as fp:
    for line in fp:
      line = line.strip(" \t\n\r")
      if line.startswith(">"):
        if description is not None:
          yield description, sequence
        description = line[1:]
        sequence = ""
      else:
        sequence += line
  yield description, sequence


def write_examples(accession_to_labels, accession_to_sequence):
  """Write examples."""
  accessions = read_txt(FLAGS.split)

  examples = []
  for accession in accessions:
    labels = accession_to_labels[accession]
    sequence = accession_to_sequence[accession]
    # Note these are not UniProt accessions, but we will store them as such
    # so that the data fields match other datasets.
    example = {
        "accession": accession,
        "sequence": sequence,
        "labels": labels,
    }
    examples.append(example)

  jsonl_utils.write(FLAGS.output, examples)


def main(unused_argv):
  # Load sequences.
  sequences_tuples = read_fasta(FLAGS.sequences)

  # Create id to sequence map.
  accession_to_sequence = {}
  for header, sequence in sequences_tuples:
    accession, meta = header.split(" ")  # pytype: disable=attribute-error
    if meta != "nrPDB":
      raise ValueError(meta)
    accession_to_sequence[accession] = sequence

  # Load EC annotations.
  annotations_rows = read_txt(FLAGS.annotations)

  accession_to_labels = {}
  # Skip 3 header rows.
  for row in annotations_rows[3:]:
    accession, ec_list = row.split("\t")
    labels = ec_list.split(",")
    accession_to_labels[accession] = labels

  write_examples(accession_to_labels, accession_to_sequence)


if __name__ == "__main__":
  app.run(main)
