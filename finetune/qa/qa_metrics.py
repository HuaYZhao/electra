# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation metrics for question-answering tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import six

import configure_finetuning
from finetune import scorer
from finetune.qa import mrqa_official_eval
from finetune.qa import squad_official_eval
from finetune.qa import squad_official_eval_v1
from model import tokenization
from util import utils

RawResult = collections.namedtuple("RawResult", [
    "unique_id", "loss", "predictions", "targets"
])


class SpanBasedQAScorer(scorer.Scorer):
    """Runs evaluation for SQuAD 1.1, SQuAD 2.0, and MRQA tasks."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, task, split,
                 v2):
        super(SpanBasedQAScorer, self).__init__()
        self._config = config
        self._task = task
        self._name = task.name
        self._split = split
        self._v2 = v2
        self._all_results = []
        self._total_loss = 0
        self._split = split
        self._eval_examples = task.get_examples(split)

    def update(self, results):
        super(SpanBasedQAScorer, self).update(results)
        self._all_results.append(
            dict(
                unique_id=results["eid"],
                loss=results["loss"],
                predictions=results["predictions"],
                targets=results["targets"],
            ))
        self._total_loss += results["loss"]

    def get_loss(self):
        return self._total_loss / len(self._all_results)

    def _get_results(self):
        self.write_predictions()
        if self._name == "squad":
            squad_official_eval.set_opts(self._config, self._split)
            squad_official_eval.main()
            return sorted(utils.load_json(
                self._config.qa_eval_file(self._name)).items())
        elif self._name == "squadv1":
            return sorted(squad_official_eval_v1.main(
                self._config, self._split).items())
        else:
            return sorted(mrqa_official_eval.main(
                self._config, self._split, self._name).items())

    def write_predictions(self):
        """Write final predictions to the json file."""
        unique_id_to_result = {}
        for result in self._all_results:
            unique_id_to_result[result["unique_id"]] = result

        results = {}
        total_loss = 0.
        for example in self._eval_examples:
            example_id = example.qas_id if "squad" in self._name else example.qid
            features = self._task.featurize(example, False, for_eval=True)

            results[example_id] = []
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature[self._name + "_eid"]]
                result['targets'] = feature[self._name + "_f1_score"]

                total_loss += (result['targets'] - result['predictions']) ** 2

                results[example_id].append(result)
        total_loss /= len(results)

        utils.write_pickle(results, self._config.f1_predict_results_file)
        utils.log(f"total_loss: {total_loss}")


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = np.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_final_text(config: configure_finetuning.FinetuningConfig, pred_text,
                   orig_text):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for i, c in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, dict(ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=config.do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if config.debug:
            utils.log(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if config.debug:
            utils.log("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if config.debug:
            utils.log("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if config.debug:
            utils.log("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text
