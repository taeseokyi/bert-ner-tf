"""BERT NER Inference."""

from __future__ import absolute_import, division, print_function

import json
import os

import tensorflow as tf
from transformers import AutoTokenizer, PreTrainedTokenizerFast, TFBartModel, TFElectraModel  # pylint: disable=unused-import

# BertNer_ORG 는 공개 학습 bert 모델이 tf 2.x 와 호환되는 것이 없다.
from tsyi_tflib.model import BertNerHF_tag  # pylint: disable=unused-import
from tsyi_tflib.official.nlp.bert.tokenization import FullSentencePieceTokenizer  # , FullTokenizer
from tsyi_tflib.tokenization import preprocess_text, convert_to_unicode, SPIECE_UNDERLINE, BasicTokenizer, mecab_bert, FullTokenizer_mecab as FullTokenizer


class Ner:

  def __init__(self, model_dir: str):
    self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
    self.label_map = self.model_config["label_map"]
    self.max_seq_length = self.model_config["max_seq_length"]
    self.label_map = {int(k): v for k, v in self.label_map.items()}

  def load_model(self, model_dir: str, model_config: str = "model_config.json"):
    model_config = os.path.join(model_dir, model_config)
    model_config = json.load(open(model_config))
    bert_config = json.load(
        open(os.path.join(model_dir, "bert_config.json")))
    model = BertNer(bert_config, "bert_config.json", 'bert_model.ckpt',
                    tf.float32, model_config['num_labels'], model_config['max_seq_length'])
    # model = BertNerHUB(tf.float32, model_config['num_labels'], model_config['max_seq_length'],
    #                  hub_module_url='https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3', hub_module_trainable=False)
    ids = tf.ones((1, 128), dtype=tf.int64)
    _ = model(ids, ids, ids, ids, input_txt=tf.constant(['']), training=False)
    model.load_weights(os.path.join(model_dir, "model.h5"))
    voacb = os.path.join(model_dir, "vocab.txt")
    tokenizer = FullTokenizer(
        vocab_file=voacb, do_lower_case=model_config["do_lower"])
    if isinstance(model, BertNerHUB):
      if model.is_albert:
        tokenizer = FullSentencePieceTokenizer(
            model.vocab_file)  # pylint: disable=no-member
      else:
        tokenizer = FullTokenizer(
            model.vocab_file, model_config["do_lower"])  # pylint: disable=no-member
    return model, tokenizer, model_config

  def tokenize(self, text: str, is_spm_model: bool = False):
    """ tokenize input"""
    # vocab.txt를 씀. 알버트일 때는 True로 변경.
    if is_spm_model:
      line = preprocess_text(text, lower=self.model_config["do_lower"])
    else:
      line = convert_to_unicode(text)
    tokens = []
    valid_positions = []

    textlist = self.tokenizer.tokenize(line)
    for word in textlist:
      # token = tokenizer.tokenize(word) # SPIECE_UNDERLINE or ## "Germany"  -- ['▁', 'G', 'erman', 'y']
      token = word
      tokens.append(token)
      if is_spm_model:
        first_is = token[1] if bytes(
            token[0], "utf-8") == SPIECE_UNDERLINE else SPIECE_UNDERLINE
      else:
        first_is = token[0] if not(token[:2] == '##') else SPIECE_UNDERLINE
      if first_is != SPIECE_UNDERLINE:
        valid_positions.append(1)
      else:
        valid_positions.append(0)
    return tokens, valid_positions

  def preprocess(self, text: str):
    """ preprocess """
    tokens, valid_positions = self.tokenize(text)
    # insert "[CLS]"
    tokens.insert(0, "[CLS]")
    valid_positions.insert(0, 1)
    # insert "[SEP]"
    tokens.append("[SEP]")
    valid_positions.append(1)
    segment_ids = []
    for _ in range(len(tokens)):
      segment_ids.append(0)
    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < self.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      valid_positions.append(0)
    return input_ids, input_mask, segment_ids, valid_positions

  def predict(self, text: str):
    input_ids, input_mask, segment_ids, valid_ids = self.preprocess(text)
    input_ids = tf.Variable([input_ids], dtype=tf.int32)
    input_mask = tf.Variable([input_mask], dtype=tf.int32)
    segment_ids = tf.Variable([segment_ids], dtype=tf.int32)
    valid_ids = tf.Variable([valid_ids], dtype=tf.int32)
    input_txt = tf.Variable([text], dtype=tf.string)
    logits = self.model(input_ids, input_mask=input_mask, input_type_ids=segment_ids,
                        valid_mask=valid_ids, training=False, input_txt=input_txt)
    logits_label = tf.argmax(logits, axis=2)
    logits_label = logits_label.numpy().tolist()[0]

    logits_confidence = [values[label].numpy()
                         for values, label in zip(logits[0], logits_label)]

    logits = []
    pos = 0
    for index, mask in enumerate(valid_ids[0]):
      if index == 0:
        continue
      if mask == 1:
        logits.append((logits_label[index-pos], logits_confidence[index-pos]))
      else:
        pos += 1
    logits.pop()

    labels = [(self.label_map[label], confidence)
              for label, confidence in logits]
    words = self.tokenizer.split_words(text)
    assert len(labels) == len(words)
    output = [{"word": word, "tag": label, "confidence": str(
        confidence)} for word, (label, confidence) in zip(words, labels)]
    return output

  def predict_list(self, text: list):
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    valid_ids_list = []
    for t in text:
      input_ids, input_mask, segment_ids, valid_ids = self.preprocess(t)
      input_ids_list.append(input_ids)
      input_mask_list.append(input_mask)
      segment_ids_list.append(segment_ids)
      valid_ids_list.append(valid_ids)

    input_ids = tf.Variable(input_ids_list, dtype=tf.int32)
    input_mask = tf.Variable(input_mask_list, dtype=tf.int32)
    segment_ids = tf.Variable(segment_ids_list, dtype=tf.int32)
    valid_ids = tf.Variable(valid_ids_list, dtype=tf.int32)
    input_txt = tf.Variable(text, dtype=tf.string)
    logits = self.model(input_ids, input_mask=input_mask, input_type_ids=segment_ids,
                        valid_mask=valid_ids, training=False, input_txt=input_txt)
    logits_label = tf.argmax(logits, axis=2)
    logits_labels = logits_label.numpy().tolist()

    outputs = []
    for i, t in enumerate(text):
      logits_label = logits_labels[i]
      logits_confidence = [values[label].numpy()
                          for values, label in zip(logits[i], logits_label)]
      logits_list = []
      pos = 0
      for index, mask in enumerate(valid_ids[i]):
        if index == 0:
          continue
        if mask == 1:
          logits_list.append((logits_label[index-pos], logits_confidence[index-pos]))
        else:
          pos += 1
      logits_list.pop()

      labels = [(self.label_map[label], confidence)
                for label, confidence in logits_list]
      words = self.tokenizer.split_words(t) # text 목록중 한개 
      assert len(labels) == len(words)
      output = [{"word": word, "tag": label, "confidence": str(
          confidence)} for word, (label, confidence) in zip(words, labels)]

      outputs.append(output)
    
    return outputs


class NerHF:

  def __init__(self, model_dir: str):
    self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
    self.hf_model_name = self.model_config["bert_model"]
    self.label_map = self.model_config["label_map"]
    self.max_seq_length = self.model_config["max_seq_length"]
    self.label_map = {int(k): v for k, v in self.label_map.items()}
    self.basic_tokenizer = BasicTokenizer(
        do_lower_case=self.model_config["do_lower"])
    self.is_spm_model = True

  def load_model(self, model_dir: str, model_config: str = "model_config.json"):
    model_config = os.path.join(model_dir, model_config)
    model_config = json.load(open(model_config))
    model = BertNerHF_tag(TFBartModel, tf.float32,
                      model_config['num_labels'], model_config['max_seq_length'], model_config["bert_model"])
    ids = tf.ones((1, model_config['max_seq_length']), dtype=tf.int64)
    _ = model(ids, ids, ids, ids, input_txt=tf.constant(['']), training=False)
    model.load_weights(os.path.join(model_dir, "model.h5"))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_config["bert_model"])
    return model, tokenizer, model_config

  def tokenize(self, text: str, is_spm_model: bool = False):
    """ tokenize input"""
    # vocab.txt를 씀. 알버트일 때는 True로 변경.
    if is_spm_model:
      line = preprocess_text(text, lower=self.model_config["do_lower"])
    else:
      line = convert_to_unicode(text)
    tokens = []
    valid_positions = []

    line = ' '.join(self.split_words(text))  # mecab 단어 분리
    textlist = self.tokenizer.tokenize(line)
    for word in textlist:
      # token = tokenizer.tokenize(word) # SPIECE_UNDERLINE or ## "Germany"  -- ['▁', 'G', 'erman', 'y']
      token = word
      tokens.append(token)
      if is_spm_model:
        first_is = "#" if bytes(
            token[0], "utf-8") == SPIECE_UNDERLINE else SPIECE_UNDERLINE
      else:
        first_is = token[0] if not(token[:2] == '##') else SPIECE_UNDERLINE
      if first_is != SPIECE_UNDERLINE:
        valid_positions.append(1)
      else:
        valid_positions.append(0)
    return tokens, valid_positions

  def preprocess(self, text: str):
    """ preprocess """
    tokens, valid_positions = self.tokenize(
        text, is_spm_model=self.is_spm_model)
    # insert "[CLS]"
    tokens.insert(0, "[CLS]")
    valid_positions.insert(0, 1)
    # insert "[SEP]"
    tokens.append("[SEP]")
    valid_positions.append(1)
    segment_ids = []
    for _ in range(len(tokens)):
      segment_ids.append(0)
    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < self.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      valid_positions.append(0)
    return input_ids, input_mask, segment_ids, valid_positions

  def predict(self, text: str):
    input_ids, input_mask, segment_ids, valid_ids = self.preprocess(text)
    input_ids = tf.constant([input_ids], dtype=tf.int32)
    input_mask = tf.constant([input_mask], dtype=tf.int32)
    segment_ids = tf.constant([segment_ids], dtype=tf.int32)
    valid_ids = tf.constant([valid_ids], dtype=tf.int32)
    input_txt = tf.constant([text], dtype=tf.string)
    logits = self.model(input_ids, input_mask=input_mask, input_type_ids=segment_ids,
                        valid_mask=valid_ids, input_txt=input_txt, training=False)
    logits_label = tf.argmax(logits, axis=2)
    logits_label = logits_label.numpy().tolist()[0]

    logits_confidence = [values[label].numpy()
                         for values, label in zip(logits[0], logits_label)]

    logits = []
    pos = 0
    for index, mask in enumerate(valid_ids[0]):
      if index == 0:
        continue
      if mask == 1:
        logits.append((logits_label[index-pos], logits_confidence[index-pos]))
      else:
        pos += 1
    logits.pop()

    labels = [(self.label_map[label], confidence)
              for label, confidence in logits]
    words = self.split_words(text)
    assert len(labels) == len(words)
    output = [{"word": word, "tag": label, "confidence": str(
        confidence)} for word, (label, confidence) in zip(words, labels)]
    return output

  def split_words(self, text):
    basic_tokens = self.basic_tokenizer.tokenize(text)
    # 이 작업을 해서 학습 입력과 동일한 mecab 분해 단위로 단어가 쪼개 져야 함.
    word_list = mecab_bert(" ".join(basic_tokens))
    return word_list


class NerHF_tag:

  def __init__(self, model_dir: str):
    self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
    self.hf_model_name = self.model_config["bert_model"]
    self.label_map = self.model_config["label_map"]
    self.max_seq_length = self.model_config["max_seq_length"]
    self.label_map = {int(k): v for k, v in self.label_map.items()}
    self.basic_tokenizer = BasicTokenizer(
        do_lower_case=self.model_config["do_lower"])
    self.is_spm_model = True

  def load_model(self, model_dir: str, model_config: str = "model_config.json"):
    model_config = os.path.join(model_dir, model_config)
    model_config = json.load(open(model_config))
    model = BertNerHF_tag(TFBartModel, tf.float32,
                      model_config['num_labels'], model_config['max_seq_length'], model_config["bert_model"])
    ids = tf.ones((1, model_config['max_seq_length']), dtype=tf.int64)
    _ = model(ids, ids, ids, ids, input_txt=tf.constant(['']), training=False)
    model.load_weights(os.path.join(model_dir, "model.h5"))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_config["bert_model"])
    return model, tokenizer, model_config

  def tokenize(self, text: str, is_spm_model: bool = False):
    """ tokenize input"""
    # vocab.txt를 씀. 알버트일 때는 True로 변경.
    if is_spm_model:
      line = preprocess_text(text, lower=self.model_config["do_lower"])
    else:
      line = convert_to_unicode(text)
    tokens = []
    valid_positions = []

    self.textlist = self.tokenizer.tokenize(line)
    for word in self.textlist:
      # token = tokenizer.tokenize(word) # SPIECE_UNDERLINE or ## "Germany"  -- ['▁', 'G', 'erman', 'y']
      token = word
      tokens.append(token)
      valid_positions.append(1)
    return tokens, valid_positions

  def preprocess(self, text: str):
    """ preprocess """
    tokens, valid_positions = self.tokenize(
        text, is_spm_model=self.is_spm_model)
    # insert "[CLS]"
    tokens.insert(0, "[CLS]")
    valid_positions.insert(0, 1)
    # insert "[SEP]"
    tokens.append("[SEP]")
    valid_positions.append(1)
    segment_ids = []
    for _ in range(len(tokens)):
      segment_ids.append(0)
    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < self.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      valid_positions.append(0)
    return input_ids, input_mask, segment_ids, valid_positions

  def predict(self, text: str):
    input_ids, input_mask, segment_ids, valid_ids = self.preprocess(text)
    input_ids = tf.constant([input_ids], dtype=tf.int32)
    input_mask = tf.constant([input_mask], dtype=tf.int32)
    segment_ids = tf.constant([segment_ids], dtype=tf.int32)
    valid_ids = tf.constant([valid_ids], dtype=tf.int32)
    input_txt = tf.constant([text], dtype=tf.string)
    logits = self.model(input_ids, input_mask=input_mask, input_type_ids=segment_ids,
                        valid_mask=valid_ids, input_txt=input_txt, training=False)
    logits_label = tf.argmax(logits, axis=2)
    logits_label = logits_label.numpy().tolist()[0]

    logits_confidence = [values[label].numpy()
                         for values, label in zip(logits[0], logits_label)]

    logits = []
    pos = 0
    for index, mask in enumerate(valid_ids[0]):
      if index == 0:
        continue
      if mask == 1:
        logits.append((logits_label[index-pos], logits_confidence[index-pos]))
      else:
        pos += 1
    logits.pop()

    labels = [(self.label_map[label], confidence)
              for label, confidence in logits]
    print(labels)
    print(self.textlist)
    print(len(labels), len(self.textlist))
    assert len(labels) == len(self.textlist)
    output = [{"word": word, "tag": label, "confidence": str(
        confidence)} for word, (label, confidence) in zip(self.textlist, labels)]
    return output
