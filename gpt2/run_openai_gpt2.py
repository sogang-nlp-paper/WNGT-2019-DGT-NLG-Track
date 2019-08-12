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
from itertools import chain
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from torch.nn import CrossEntropyLoss, BCELoss

from pytorch_transformers import (GPT2Tokenizer,
                                  AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME)

from modeling_gpt2 import GPT2EntityEncoderLMModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

DEC_MAX_LEN = 800  # TODO gpt2 model is limited 1024, need to reduce input record size
n_record = 602


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


def encode_dataset(args, device, tokenizer, pad_token, _type="train", _index=False):
    logger.info("Encoding {} dataset...".format(_type))
    if os.path.exists(os.path.join(args.dataset_path, "{}.pt".format(_type))) and \
            os.path.exists(os.path.join(args.dataset_path, "{}_vocab_mask.pt".format(_type))):
        tensor_dataset = torch.load(os.path.join(args.dataset_path, "{}.pt".format(_type)))
        record_vocab_mask = torch.load(os.path.join(args.dataset_path, "{}_vocab_mask.pt".format(_type)))
    else:
        raw_src, raw_tgt = load_rotowire_dataset(args.dataset_path, _type, index=_index)
        datasets = (raw_src, raw_tgt)
        src, tgt = tokenize_and_encode(datasets, tokenizer)
        tensor_dataset, record_vocab_mask = pre_process_datasets(device, src, tgt, pad_token, tokenizer.vocab_size)
        logger.info("save {}.pt binary...".format(_type))
        torch.save(tensor_dataset, os.path.join(args.dataset_path, "{}.pt".format(_type)))
        logger.info("save {}_vocab_mask.pt binary...".format(_type))
        torch.save(record_vocab_mask, os.path.join(args.dataset_path, "{}_vocab_mask.pt".format(_type)))
    return tensor_dataset, record_vocab_mask.float()


def load_rotowire_dataset(dataset_path, _type="train", index=False):
    src_path = os.path.join(dataset_path, 'src_'+_type+'.txt')
    tgt_path = os.path.join(dataset_path, 'tgt_'+_type+'.txt')

    src_data = []
    with open(src_path, encoding='utf_8') as f:
        for line in f.readlines():
            src_data.append([[x for x in t.split('￨')] for t in line.split()])

    with open(tgt_path, encoding='utf_8') as f:
        tgt_data = [line for line in f.readlines()]
    assert len(src_data) == len(tgt_data)
    if index:
        assert type(index) == int
        return src_data[index:index+1], tgt_data[index:index+1]
    return src_data, tgt_data


def pre_process_datasets(device, src, tgt, pad_token, vocab_size):
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

    record_vocab_mask = torch.zeros(vocab_size).to(device)
    for record in src:
        for entity in record:
            for i, elem in enumerate(entity):
                for e in elem:
                    record_vocab_mask[int(e)] = 1
                pad_size = src_max_len - len(elem)
                if pad_size > 0:
                    entity[i] = elem + ([0] * pad_size)
                assert len(entity[i]) == src_max_len

    # padding for tgt
    tgt_max_len = min(max([len(summary) for summary in tgt]), DEC_MAX_LEN)

    # for copy supervision
    copy_labels = []
    for i, summary in enumerate(tgt):
        summary = summary + [pad_token]
        pad_size = tgt_max_len - len(summary)
        tgt[i] = summary + ([pad_token] * pad_size) if pad_size > 0 else summary[:tgt_max_len]
        copy_labels.append([record_vocab_mask[x].item() for x in tgt[i]])
        assert len(tgt[i]) == tgt_max_len

    src = torch.tensor(src, dtype=torch.long).to(device)  # (N, 602, 4, src_max_len)
    tgt = torch.tensor(tgt, dtype=torch.long).to(device)  # (N, tgt_max_len)

    copy_labels = torch.tensor(copy_labels, dtype=torch.float).to(device)

    return TensorDataset(src, tgt, copy_labels), record_vocab_mask.to(device)


def validate(model, device, n_gpu, eval_dataloader, vocab_mask):
    model.eval()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        record_ids, lm_labels, copy_labels = batch

        inputs = {'record_ids': record_ids, 'labels': lm_labels, 'src_vocab_mask': vocab_mask}
        outputs = model(**inputs)
        lm_logits = outputs[0][:, :-1, :].contiguous()
        labels = lm_labels.contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss_lm = loss_fct(lm_logits.view(-1, lm_logits.size(-1)),
                           labels.view(-1))
        gate = outputs[-1]
        loss_gate = BCELoss()(gate.view(-1), copy_labels.view(-1))
        loss = (loss_lm + loss_gate) / 2
        if n_gpu > 1:
            loss = loss.mean()
        tr_loss += loss.item()
        nb_tr_steps += 1
    dev_loss = tr_loss / nb_tr_steps
    logger.info("loss (dev set): {:.2e}".format(dev_loss))
    return dev_loss


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
    parser.add_argument("--generate_model_file", default="pytorch_model.bin", type=str)
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
    parser.add_argument('--check_point', type=str, default='')

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

    # Load and encode the datasets
    if not args.dataset_path:
        raise FileNotFoundError("data path not found")
        # roc_stories = cached_path(ROCSTORIES_URL)

    if args.do_train or args.do_eval:
        # special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
        model = GPT2EntityEncoderLMModel.from_pretrained(args.model_name)
        model.to(device)

        train_data, train_vocab_mask = encode_dataset(args, device, tokenizer, pad_token, _type="train")
        eval_data, eval_vocab_mask = encode_dataset(args, device, tokenizer, pad_token, _type="valid")

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # eval_sampler = SequentialSampler(eval_data)
        eval_sampler = RandomSampler(eval_data, replacement=True, num_samples=100)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        sample_data, sample_vocab_mask = encode_dataset(args, device, tokenizer, pad_token, _type="valid_temp")
        sample_dataloader = DataLoader(sample_data, sampler=SequentialSampler(sample_data), batch_size=1)

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

        start = 0
        if args.check_point:
            check_point_file = os.path.join(args.output_dir, args.check_point)
            state_dict = torch.load(check_point_file)
            model.load_state_dict(state_dict)
            logger.info(" start training from check point %s" % check_point_file)
            start = int(args.check_point.split("-")[1]) + 1

        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        for ep in trange(start, int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                record_ids, lm_labels, copy_labels = batch

                inputs = {'record_ids': record_ids, 'labels': lm_labels, 'src_vocab_mask': train_vocab_mask}
                outputs = model(**inputs)
                lm_logits = outputs[0][:, :-1, :].contiguous()
                labels = lm_labels.contiguous()
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                loss_lm = loss_fct(lm_logits.view(-1, lm_logits.size(-1)),
                                   labels.view(-1))
                # loss for copy supervision
                gate = outputs[-1][:, :-1, :].contiguous()
                loss_gate = BCELoss()(gate.view(-1), copy_labels.view(-1))
                loss = loss_lm + loss_gate
                if n_gpu > 1:
                    loss = loss.mean()
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                nb_tr_steps += 1
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.4f}".format(exp_average_loss, optimizer.defaults["lr"])
                # end of batch
            # end of all batch
            # early stopping
            if args.early_stop and ep > 30 and (ep+1) % 4 == 0:
                logger.info("Epoch %s Validating ..." % str(ep+1))
                eval_loss = validate(model, device, n_gpu, eval_dataloader, eval_vocab_mask)
                if eval_loss > pre_loss:
                    tolerance -= 1
                    stop_flag = True if tolerance == 0 else False
                else:
                    tolerance = args.early_stop_tolerance
                pre_loss = eval_loss
            if args.early_stop and stop_flag:
                logger.info("Training finished after not improving. Early Stop!")
                # break
            # save model
            if args.do_save and (ep+1) % 10 == 0:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file + "-" + str(ep+1))
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)
                logger.info("Save model %s" % output_model_file + "-" + str(ep+1))
                logger.info("Sample summary by %s" % output_model_file + "-" + str(ep+1))
                for record_ids, _, _ in sample_dataloader:
                    summary = generate_summary(model, record_ids, tokenizer, sample_vocab_mask)
                    logger.info(summary.encode('utf-8').decode('utf-8'))
            # end of epoch

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

    if args.do_generate:
        model = GPT2EntityEncoderLMModel.from_pretrained(args.output_dir)
        state_dict = torch.load(os.path.join(args.output_dir, args.generate_model_file))
        model.load_state_dict(state_dict)
        model.to(device)

        # test_data = encode_dataset(args, device, tokenizer, pad_token, _type="test", _index=53)
        test_data, test_vocab_mask = encode_dataset(args, device, tokenizer, pad_token, _type="test")
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)
        model.eval()

        logger.info("***** Generate summary %s *****" % (args.output_dir+"/pred.txt"))
        with open(os.path.join(args.output_dir, "pred.txt"), "w", encoding="utf-8") as f:
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                record_ids, lm_labels, _ = batch
                summary = generate_summary(model, record_ids, tokenizer, test_vocab_mask)
                print(summary.encode('utf-8').decode('utf-8'))
                f.write(summary.encode('utf-8').decode('utf-8') + "\n")

            # test
            # print(tokenizer.decode(lm_labels[0].tolist()))
            # print("---------------------------------------------------------------")
            # print(tokenizer.decode(summary_ids[0].tolist()))
            # print("")


def generate_summary(model, record_ids, tokenizer, vocab_mask):
    summary_ids = None
    with torch.no_grad():
        for i in range(DEC_MAX_LEN):
            outputs = model(record_ids, summary_ids=summary_ids, src_vocab_mask=vocab_mask)
            next_token_logits = outputs[0][:, -1, :]
            next_token_id = torch.argmax(next_token_logits).view(-1, 1)
            summary_ids = next_token_id if i == 0 else torch.cat((summary_ids, next_token_id), dim=1)
            next_token = tokenizer.convert_ids_to_tokens(next_token_id[0].tolist())[0]
            if next_token == '<|endoftext|>' or next_token is None:
                break
    summary = tokenizer.decode(summary_ids[0].tolist(), '<|endoftext|>')
    return summary


if __name__ == '__main__':
    main()
