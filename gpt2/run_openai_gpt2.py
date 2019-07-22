# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
          --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2Tokenizer, GPT2Config,
                                  # GPT2LMHeadModel,
                                  AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME)

from modeling_gpt2 import GPT2EntityEncoderLMModel

# ROCSTORIES_URL = "https://s3.amazonaws.com/datasets.huggingface.co/ROCStories.tar.gz"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def load_rotowire_dataset(dataset_path, _type="train"):
    src_path = os.path.join(dataset_path, 'src_'+_type+'.txt')
    tgt_path = os.path.join(dataset_path, 'tgt_'+_type+'.txt')

    src_data = []
    with open(src_path, encoding='utf_8') as f:
        for line in f.readlines():
            src_data.append([[x for x in t.split('￨')] for t in line.split()])

    with open(tgt_path, encoding='utf_8') as f:
        tgt_data = [line for line in f.readlines()]
    assert len(src_data) == len(tgt_data)
    return src_data, tgt_data


def pre_process_datasets(src, tgt, device):
    """ pre-process Rotowire dataset
    set padding for WPE
    'F|Stephen Curry|START_POSITION|HOME'
    => [['F'], ['Ste', '##phe', '##n', 'Curry'], ['START', 'POSITION'], ['HOME']]
    => [[37, 0, 0, 0, 0, 0], ... ]
    """
    # padding for src
    src_max_len = 0
    for record in src:
        for entity in record:
            src_max_len = max(src_max_len, max([len(e) for e in entity]))

    for record in src:
        for entity in record:
            for i, elem in enumerate(entity):
                pad_size = src_max_len - len(elem)
                if pad_size > 0:
                    entity[i] = elem + ([0] * pad_size)
                assert len(entity[i]) == src_max_len

    # padding for tgt
    tgt_max_len = max([len(summary) for summary in tgt])
    for i, summary in enumerate(tgt):
        pad_size = tgt_max_len - len(summary)
        if pad_size > 0:
            tgt[i] = summary + ([0] * pad_size)
        assert len(tgt[i]) == tgt_max_len

    src = torch.tensor(src, dtype=torch.long).to(device)  # (N, 602, 4, src_max_len)
    tgt = torch.tensor(tgt, dtype=torch.long).to(device)  # (N, tgt_max_len)

    return TensorDataset(src, tgt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument('--train_dataset', type=str, default='')
    # parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the Rotowire dataset
    # special_tokens = ['_start_', '_delimiter_', '_classify_']
    config = GPT2Config.from_pretrained(args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    # special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = GPT2EntityEncoderLMModel.from_pretrained(args.model_name)
    model.to(device)

    # Load and encode the datasets
    if not args.dataset_path:
        raise FileNotFoundError("data path not found")
        # roc_stories = cached_path(ROCSTORIES_URL)

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        # elif isinstance(obj, list):
        #     return list(tokenize_and_encode(e) for e in obj)
        return list(tokenize_and_encode(o) for o in obj)
    logger.info("Encoding dataset...")

    # FIXME when you train
    # train_src, train_tgt = load_rotowire_dataset(args.dataset_path, "train")
    train_src, train_tgt = load_rotowire_dataset(args.dataset_path, "valid_temp")
    # eval_src, eval_tgt = load_rotowire_dataset(args.dataset_path, "valid")
    eval_src, eval_tgt = train_src, train_tgt

    datasets = (train_src, train_tgt, eval_src, eval_tgt)
    train_src, train_tgt, eval_src, eval_tgt = tokenize_and_encode(datasets)

    # Compute the max input length for the Transformer
    # max_length = model.config.n_positions - 2  # n.positions = 1024
    # input_length = max([len(cont[:max_length]) + 2 for dataset in encoded_datasets for cont in dataset])
    # input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model

    # Prepare inputs tensors and dataloaders
    # train_tensor_dataset = pre_process_datasets(train_src, train_tgt)
    # eval_tensor_dataset = pre_process_datasets(eval_src, eval_tgt)

    train_data = pre_process_datasets(train_src, train_tgt, device)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = pre_process_datasets(eval_src, eval_tgt, device)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters,
                               lr=args.learning_rate,
                               # warmup=args.warmup_proportion,
                               # max_grad_norm=args.max_grad_norm,
                               weight_decay=args.weight_decay)
                               # t_total=num_train_optimization_steps)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels = batch
                losses = model(input_ids, label=lm_labels)
                loss = args.lm_coef * losses[0] + losses[1]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.output_dir)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
        model.to(device)

    if args.do_eval:
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            with torch.no_grad():
                _, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                _, mc_logits = model(input_ids, mc_token_ids)

            mc_logits = mc_logits.detach().cpu().numpy()
            mc_labels = mc_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

            eval_loss += mc_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        train_loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'train_loss': train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == '__main__':
    main()
