#!/usr/bin/env python

import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer

from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel

from utils.data import load_data

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embeddings(sentences, tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)


def save_task_prompt_weights(task, tokenizer, model, train_data, outdir):
    prompts = None
    raise NotImplementedError
    with open(os.path.join(outdir, task + '.pkl'), 'rb') as f:
        pkl.dump(f, prompts)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    args = Namespace()
    args.gpt2 = 'gpt2-large'
    args.checkpoint = 'checkpoints/metaicl/hr_to_lr/model.pt'
    args.global_step = None
    args.use_demonstrations = False
    args.do_zeroshot = True
    args.k = 0
    args.total_data_size = 200
    # args.test_batch_size = 8
    args.test_batch_size = 16
    args.method = 'direct'
    args.task = 'custom'
    args.unseen_domain_only = False
    args.dataset = None
    args.split = 'dev'
    args.is_null = False
    # args.out_dir = None
    # args.out_dir = 'results/gpt2_uncertainty_sampling'
    args.out_dir = 'results/gpt2finetuned_uncertainty_sampling'
    args.seed = '100'
    args.use_random_english_words = False
    args.use_calibration = False
    # args.num_prompt_samples = 32
    # args.ks = [0, 1, 2, 4, 8, 16, 32]
    # args.trim_dev_data = 32
    args.gpt2s = 'gpt2-large'
    args.checkpoints = 'checkpoints/metaicl/hr_to_lr/model.pt'
    # args.gpt2s = 'gpt2-large'
    # args.checkpoints = 'gpt2-large'
    # args.gpt2s = 'gpt-j-6B'
    # args.checkpoints = 'gpt-j-6B'
    args.prompt_with_random_tasks = True
    args.sampling_weights_dir = args.out_dir
    args.c = 1
    if not os.path.exists(args.sampling_weights_dir):
        os.mkdir(args.sampling_weights_dir)

    # In[4]:

    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"
    handlers = [logging.StreamHandler()]
    # if args.log_file is not None:
    #     handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    from tqdm import tqdm
    add_newlines = True

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data

    max_length_per_example = 256
    max_length = 256
    if args.use_demonstrations:
        orig_max_length = max_length
        if args.do_zeroshot:
            max_length = min(max_length * args.k, 1024)
        else:
            max_length = min(max_length * args.k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(logger, tokenizer, args.method, args.use_demonstrations, args.k,
                               max_length, max_length_per_example)

    # In[ ]:

    seeds = args.seed.split(",")

    for seed in seeds:
        ### data ...
        train_data = load_data(args.task, "train", args.total_data_size, seed=seed, config_split=config_split,
                               datasets=None if args.dataset is None else args.dataset.split(","))

        if args.use_random_english_words:
            from english_words import english_words_set

            english_words_set = sorted(english_words_set)
            np.random.seed(int(seed))

        train_counter = Counter()
        dev_counter = Counter()
        for dp in train_data:
            train_counter[dp["task"]] += 1
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))

        logger.info("%s on %s (%d train)" % (args.method, args.task, len(train_counter)))

        for test_task in tqdm(list(train_counter)):
            if os.path.exists(os.path.join(args.sampling_weights_dir, test_task + '.pkl')):
                continue
            task_train_data = [dp for dp in train_data if dp["task"] == test_task]

            # if args.sampling_weights_dir is not None:
            #     assert not args.prompt_with_random_tasks
            #     with open(os.path.join(args.sampling_weights_dir, test_task + '.pkl'), 'rb') as f:
            #         task_weights = pkl.load(f)
            # else:
            #     task_weights = None

            # assert len(task_dev_data) > 0
            assert not args.use_demonstrations or len(task_train_data) == args.total_data_size, \
                (args.use_demonstrations, len(task_train_data), args.total_data_size)

            config_file = "config/tasks/{}.json".format(test_task)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"] == "classification"
            if is_classification:
                options = task_train_data[0]["options"]
                assert np.all([d["options"] == options for d in task_train_data])
            if args.use_random_english_words:
                # create a mapping
                options = task_train_data[0]["options"]
                mapping = {option: np.random.choice(english_words_set) for option in options}
                new_options = list(mapping.values())
                for dp_idx, dp in enumerate(task_train_data):
                    assert dp["output"] in options, (dp, options)
                    task_train_data[dp_idx]["output"] = mapping[dp["output"]]
                    task_train_data[dp_idx]["options"] = new_options
                for dp_idx, dp in enumerate(task_train_data):
                    assert dp["output"] in options, (dp, options)
                    task_train_data[dp_idx]["output"] = mapping[dp["output"]]
                    task_train_data[dp_idx]["options"] = new_options
            save_task_prompt_weights(
                test_task, metaicl_data, tokenizer, model, task_train_data, add_newlines, args)
