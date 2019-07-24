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

from torch.nn import CrossEntropyLoss

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


def tokenize_and_encode(obj, tokenizer):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    # elif isinstance(obj, list):
    #     return list(tokenize_and_encode(e) for e in obj)
    return list(tokenize_and_encode(o, tokenizer) for o in obj)


def encode_dataset(args, device, tokenizer, pad_token, _type="train"):
    logger.info("Encoding dataset...")
    # FIXME when you train
    src, tgt = load_rotowire_dataset(args.dataset_path, _type)
    if os.path.exists(os.path.join(args.dataset_path, _type + ".pt")):
        tensor_dataset = torch.load(os.path.join(args.dataset_path, _type + ".pt"))
        logger.info("load %s" % os.path.join(args.dataset_path, _type + ".pt"))
    else:
        datasets = (src, tgt)
        src, tgt = tokenize_and_encode(datasets, tokenizer)
        tensor_dataset = pre_process_datasets(device, src, tgt, pad_token)
        torch.save(tensor_dataset, os.path.join(args.dataset_path, _type + ".pt"))
        logger.info("save %s" % os.path.join(args.dataset_path, _type + ".pt"))
    return tensor_dataset


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


def pre_process_datasets(device, src, tgt, pad_token):
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
    # tgt_max_len = tgt_length
    for i, summary in enumerate(tgt):
        summary = summary + [pad_token]
        pad_size = tgt_max_len - len(summary)
        tgt[i] = summary + ([pad_token] * pad_size) if pad_size > 0 else summary[:tgt_max_len]
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
    parser.add_argument("--do_generate", action='store_true')
    parser.add_argument("--do_save", action='store_true')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
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
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--early_stop_tolerance', type=int, default=3)

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

    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the Rotowire dataset

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    pad_token = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    # special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = GPT2EntityEncoderLMModel.from_pretrained(args.model_name)
    model.to(device)

    # Load and encode the datasets
    if not args.dataset_path:
        raise FileNotFoundError("data path not found")
        # roc_stories = cached_path(ROCSTORIES_URL)

    if args.do_train:
        # FIXME when you train
        # train_data = encode_dataset(args, device, tokenizer, pad_token, _type="train")
        train_data = encode_dataset(args, device, tokenizer, pad_token, _type="valid_temp")

    # eval_data = encode_dataset(args, device, tokenizer, pad_token, _type="valid")
    eval_data = encode_dataset(args, device, tokenizer, pad_token, _type="valid_temp")

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

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

    pre_loss = 9999999
    stop_flag = False
    tolerance = args.early_stop_tolerance

    if args.do_train:
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                record_ids, lm_labels = batch

                # sequence
                seq_len = lm_labels.size(1)
                summary_ids = None
                tmp_loss = 0
                for i in range(seq_len):
                    inputs = {'record_ids': record_ids, 'summary_ids': summary_ids}

                    outputs = model(**inputs, labels=lm_labels)
                    next_token_logits = outputs[0][:, -1, :]
                    next_token_label = lm_labels[:, i]
                    loss_fct = CrossEntropyLoss(ignore_index=-1)
                    if args.train_batch_size == 1:
                        tmp_loss = tmp_loss + loss_fct(next_token_logits, next_token_label)
                        next_token = lm_labels[:, i].unsqueeze(0)
                    else:
                        tmp_loss = tmp_loss + loss_fct(next_token_logits.squeeze(), next_token_label)
                        next_token = lm_labels[:, i]
                    summary_ids = next_token if i == 0 else torch.cat((summary_ids, next_token), dim=1)

                loss = tmp_loss / seq_len
                if n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.4f} lr: {:.4f}".format(exp_average_loss, optimizer.defaults["lr"])

            # early stopping
            if tr_loss > pre_loss:
                tolerance -= 1
                stop_flag = True if tolerance == 0 else False
            else:
                tolerance = args.early_stop_tolerance
            pre_loss = tr_loss
            if args.early_stop and stop_flag:
                logger.info("Training finished after not improving. Early Stop!")
                break
            # save model
            if (ep+1) % 30 == 0:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file + "-" + str(ep+1))
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

    # Save a trained model
    if args.do_save:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

    if args.do_eval:
        # Load a trained model and vocabulary that you have fine-tuned
        model = GPT2EntityEncoderLMModel.from_pretrained(args.output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
        model.to(device)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, lm_labels = batch
            with torch.no_grad():
                lm_loss = model(input_ids, labels=lm_labels)
                lm_logits, _ = model(input_ids)

            lm_logits = lm_logits.detach().cpu().numpy()
            lm_labels = lm_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(lm_logits, lm_labels)

            eval_loss += lm_loss.mean().item()
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

    if args.do_generate:
        # Load a trained model and vocabulary that you have fine-tuned
        model = GPT2EntityEncoderLMModel.from_pretrained(args.output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
        model.to(device)
        # test_data = encode_dataset(args, device, tokenizer, pad_token, _type="test")
        test_data = encode_dataset(args, device, tokenizer, pad_token, _type="valid_temp")
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)
        model.eval()

        logger.info("***** Generate summary %s *****" % (args.output_dir+"pred.txt"))
        for batch in test_dataloader:
            with open(os.path.join(args.output_dir, "pred.txt"), "w") as f:
                batch = tuple(t.to(device) for t in batch)
                record_ids, lm_labels = batch
                summary_ids = None
                summary = []
                with torch.no_grad():
                    outputs = model(record_ids, summary_ids=summary_ids)
                    next_token_logits = outputs[0][:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=1)
                    next_token = tokenizer.convert_ids_to_tokens(next_token_id)
                    summary.append(next_token)
                    if next_token == '<|endoftext|>':
                        break

                f.write(tokenizer.decode(summary) + "\n")

            print(tokenizer.decode(lm_labels[0]))
            print("---------------------------------------------------------------")
            print(tokenizer.decode(summary))
            print("")


if __name__ == '__main__':
    main()
