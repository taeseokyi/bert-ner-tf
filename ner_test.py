import os
import math
import json
import argparse
import numpy as np
from fastprogress import master_bar, progress_bar
from seqeval.metrics import classification_report

import tsyi_tflib
from tsyi_tflib.run_ner_hf_tag import NerProcessor, convert_examples_to_features, logger
from tsyi_tflib.official.common import distribute_utils
from tsyi_tflib.optimization import AdamWeightDecay, WarmUp
from tsyi_tflib.model import BertNerHF_tag

import tensorflow as tf
from transformers import AutoTokenizer, PreTrainedTokenizerFast, TFBartModel, TFElectraModel  # Registers the ops. pylint:disable=unused-import

def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--data_dir",
                      default=f"{tsyi_tflib.__path__[0]}/data/tech_name_tag",
                      type=str,
                      required=False,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--output_dir",
                      default="out_base_ner_kobart",
                      type=str,
                      required=False,
                      help="The output directory where the model predictions and checkpoints will be written.")
  parser.add_argument("--hf_model_name", default="gogamza/kobart-base-v2", type=str, 
                      required=False,
                      help="String that represents tensorflow hugging face's model name.")
  parser.add_argument("--do_spm_model", action='store_true',
                      help="Tokenizer's type.")
  # Other parameters
  parser.add_argument("--max_seq_length",
                      default=512,
                      type=int,
                      help="The maximum total input sequence length after WordPiece tokenization. \n"
                           "Sequences longer than this will be truncated, and sequences shorter \n"
                           "than this will be padded.")
  parser.add_argument("--do_train", default=True,
                      action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval", default=True,
                      action='store_true',
                      help="Whether to run eval on the dev/test set.")
  parser.add_argument("--eval_on",
                      default="dev",
                      type=str,
                      help="Evaluation set, dev: Development, test: Test")
  parser.add_argument("--do_lower_case",
                      action='store_true',
                      help="Set this flag if you are using an uncased model.")
  parser.add_argument("--train_batch_size",
                      default=2,
                      type=int,
                      help="Total batch size for training.")
  parser.add_argument("--eval_batch_size",
                      default=64,
                      type=int,
                      help="Total batch size for eval.")
  parser.add_argument("--learning_rate",
                      default=5e-5,
                      type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs",
                      default=3,
                      type=int,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion",
                      default=0.1,
                      type=float,
                      help="Proportion of training to perform linear learning rate warmup for. "
                           "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--weight_decay", default=0.01, type=float,
                      help="Weight deay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                      help="Epsilon for Adam optimizer.")
  parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help="random seed for initialization")
  # training stratergy arguments
  parser.add_argument("--multi_gpu",
                      action='store_true',
                      help="Set this flag to enable multi-gpu training using MirroredStrategy."
                           "Single gpu training")
  parser.add_argument("--gpus", default='0', type=str,
                      help="Comma separated list of gpus devices."
                      "For Single gpu pass the gpu id.Default '0' GPU"
                      "For Multi gpu,if gpus not specified all the available gpus will be used")
  parser.add_argument("--tpu", default=None, type=str,
                      help="Optional. String that represents TPU to connect to. Must not be None if `distribution_strategy` is set to `tpu`")

  parser.add_argument("--init_hf_model", default=None, type=str,
                      help="Optional. String that represents a pre-trained file in the hugging-face model directory for fine-tuning.")
  args = parser.parse_args()
  print(args.data_dir)
  print(args.hf_model_name)
  print(args.do_spm_model)
  print(args.output_dir)
  print(args.max_seq_length)
  print(args.do_train)
  print(args.num_train_epochs)
  print(args.do_eval)
  print(args.eval_on)
  print(args.train_batch_size)

  processor = NerProcessor()
  label_list = processor.get_labels()
  num_labels = len(label_list) + 1 # 0 is label padding. 따라서 num_labels 는 실제 갯수 보다 1 큼.

  if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError(
        "Output directory ({}) already exists and is not empty.".format(args.output_dir))
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  if args.do_train:
    # using hubhugging face's a transformer model name which is a pretrained language model.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.hf_model_name)

  if args.multi_gpu:
    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy="mirrored",
        num_gpus=len(args.gpus.split(',')))
  else:
    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy="one_device",
        num_gpus=1)

  train_examples = None
  optimizer = None
  num_train_optimization_steps = 0
  ner = None
  if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size) * args.num_train_epochs
    warmup_steps = int(args.warmup_proportion *
                       num_train_optimization_steps)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.learning_rate,
                                                                     decay_steps=num_train_optimization_steps, end_learning_rate=0.0)
    if warmup_steps:
      learning_rate_fn = WarmUp(initial_learning_rate=args.learning_rate,
                                decay_schedule_fn=learning_rate_fn,
                                warmup_steps=warmup_steps)
    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=args.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=args.adam_epsilon,
        exclude_from_weight_decay=['layer_norm', 'bias'])

    with strategy.scope():
      ner = BertNerHF_tag(TFBartModel, tf.float32, num_labels, args.max_seq_length, args.hf_model_name, args.init_hf_model)
      loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
          reduction=tf.keras.losses.Reduction.NONE)

  label_map = {i: label for i, label in enumerate(label_list, 1)}
  if args.do_train:
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, is_spm_model=args.do_spm_model, do_lower_case=args.do_lower_case)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    all_input_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_ids for f in train_features]))
    all_input_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_mask for f in train_features]))
    all_segment_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.segment_ids for f in train_features]))
    all_valid_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.valid_ids for f in train_features]))
    all_label_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_mask for f in train_features]))

    all_label_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_id for f in train_features]))
    all_input_txt = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_txt for f in train_features]))

    # Dataset using tf.data
    train_data = tf.data.Dataset.zip(
        (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids, all_label_mask, all_input_txt))
    shuffled_train_data = train_data.shuffle(buffer_size=int(len(train_features) * 0.1),
                                             seed=args.seed,
                                             reshuffle_each_iteration=True)
    batched_train_data = shuffled_train_data.batch(args.train_batch_size)
    # Distributed dataset
    #dist_dataset = strategy.experimental_distribute_dataset(batched_train_data)
    dist_dataset = batched_train_data

    loss_metric = tf.keras.metrics.Mean()

    epoch_bar = master_bar(range(args.num_train_epochs))
    pb_max_len = math.ceil(
        float(len(train_features))/float(args.train_batch_size))

    def train_step(input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask, input_txt):
      def step_fn(input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask, input_txt):

        with tf.GradientTape() as tape:
          logits = ner(input_ids, input_mask, segment_ids,
                       valid_ids, input_txt=input_txt, training=True)
          label_mask = tf.reshape(label_mask, (-1,))
          logits = tf.reshape(logits, (-1, num_labels))
          logits_masked = tf.boolean_mask(logits, label_mask)
          label_ids = tf.reshape(label_ids, (-1,))
          label_ids_masked = tf.boolean_mask(label_ids, label_mask)
          cross_entropy = loss_fct(label_ids_masked, logits_masked)
          loss = tf.reduce_sum(cross_entropy) * (1.0 / args.train_batch_size)
        grads = tape.gradient(loss, ner.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, ner.trainable_variables)))
        return cross_entropy

      per_example_losses = strategy.run(step_fn,
                                        args=(input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask, input_txt))
      mean_loss = strategy.reduce(
          tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
      return mean_loss

    for epoch in epoch_bar:  # pylint:disable=unused-variable
      with strategy.scope():
        for (input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask, input_txt) in progress_bar(dist_dataset, total=pb_max_len, parent=epoch_bar):
          loss = train_step(input_ids, input_mask, segment_ids,
                            valid_ids, label_ids, label_mask, input_txt)
          loss_metric(loss)
          epoch_bar.child.comment = f'loss : {loss_metric.result()}'
      loss_metric.reset_states()

    # model weight save
    ner.save_weights(os.path.join(args.output_dir, "model.h5"))

    # copy vocab to output_dir
    # copy bert config to output_dir

    ner.config.to_json_file(os.path.join(
        args.output_dir, 'transformer_encoder.config'))
    #tokenizer.save_vocabulary(args.output_dir)

    # save label_map and max_seq_length of trained model
    model_config = {"bert_model": args.hf_model_name, "do_lower": args.do_lower_case,
                    "max_seq_length": args.max_seq_length, "num_labels": num_labels,
                    "label_map": label_map}
    json.dump(model_config, open(os.path.join(
        args.output_dir, "model_config.json"), "w"), indent=4)

  if args.do_eval:
    # load tokenizer
    # config is json object, not str.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.hf_model_name)

    ner = BertNerHF_tag(TFBartModel, tf.float32, num_labels, args.max_seq_length, args.hf_model_name)

    ids = tf.ones((1, args.max_seq_length), dtype=tf.int64)
    _ = ner(ids, ids, ids, ids, input_txt=tf.constant(
        ['']), training=False)  # 배치자료가 입력되는 것으로 값을 제공
    ner.load_weights(os.path.join(
        args.output_dir, "model.h5"))  # 학습결과 모델 새로 로딩

    # load test or development set based on argsK
    if args.eval_on == "dev":
      eval_examples = processor.get_dev_examples(args.data_dir)
    elif args.eval_on == "test":
      eval_examples = processor.get_test_examples(args.data_dir)

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, is_spm_model=args.do_spm_model, do_lower_case=args.do_lower_case)
    logger.info("***** Running evalution *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_ids for f in eval_features]))
    all_input_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_mask for f in eval_features]))
    all_segment_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.segment_ids for f in eval_features]))
    all_valid_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.valid_ids for f in eval_features]))

    all_label_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_id for f in eval_features]))
    all_input_txt = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_txt for f in eval_features]))

    eval_data = tf.data.Dataset.zip(
        (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids, all_input_txt))
    batched_eval_data = eval_data.batch(args.eval_batch_size)

    loss_metric = tf.keras.metrics.Mean()
    epoch_bar = master_bar(range(1))
    pb_max_len = math.ceil(
        float(len(eval_features))/float(args.eval_batch_size))

    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for epoch in epoch_bar:
      for (input_ids, input_mask, segment_ids, valid_ids, label_ids, input_txt) in progress_bar(batched_eval_data, total=pb_max_len, parent=epoch_bar):
        logits = ner(input_ids, input_mask,
                     segment_ids, valid_ids, training=False, input_txt=input_txt)
        logits = tf.argmax(logits, axis=2)
        for i, label in enumerate(label_ids):
          temp_1 = []
          temp_2 = []
          for j, _ in enumerate(label):
            if j == 0:
              continue
            elif label_map[label_ids[i][j].numpy()] == '[SEP]':  # 끝나는 조건 [SEP] 이면 탈출
              y_true.append(temp_1)
              y_pred.append(temp_2)
              break
            else:
              temp_1.append(label_map[label_ids[i][j].numpy()])
              temp_2.append(label_map[logits[i][j].numpy()])
    report = classification_report(y_true, y_pred, digits=4)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
      logger.info("***** Eval results *****")
      logger.info("\n%s", report)
      writer.write(report)


if __name__ == "__main__":
  main()

# pylint: disable=pointless-string-statement
'''

python ner_test.py  \
  --data_dir=data/tech_name_tag  \
  --hf_model_name=monologg/koelectra-small-v2-discriminator  \
  --output_dir=out_base_ner_hf  \
  --max_seq_length=512  \
  --do_train  \
  --num_train_epochs=3  \
  --do_eval  \
  --eval_on=dev  \
  --train_batch_size=4

  --multi_gpu \
  --gpus=0,1

python ner_test.py  \
  --data_dir=data/tech_name_tag  \
  --hf_model_name=gogamza/kobart-base-v2  \
  --do_spm_model  \
  --output_dir=out_base_ner_kobart  \
  --max_seq_length=512  \
  --do_train  \
  --num_train_epochs=3  \
  --do_eval  \
  --eval_on=dev  \
  --train_batch_size=2

  --multi_gpu \
  --gpus=0,1
'''
