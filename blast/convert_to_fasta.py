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

"""Convert from examples jsonl to fasta format.

Example of protein in fasta file format:

>accession="Q0WJ82"
MEKTQSVFIRFIVNGSLVKQILIGLVAGIVLALVST...
"""

from absl import app
from absl import flags
import tensorflow as tf

from common import jsonl_utils


FLAGS = flags.FLAGS


flags.DEFINE_string("input", "", "Path to examples.")

flags.DEFINE_string("output", "", "Fasta output file.")


def main(unused_argv):
  examples = jsonl_utils.read(FLAGS.input)
  with tf.io.gfile.GFile(FLAGS.output, "w") as fp:
    for example in examples:
      fp.write(f'>accession="{example["accession"]}"\n')
      fp.write(f'{example["sequence"]}\n')


if __name__ == "__main__":
  app.run(main)
