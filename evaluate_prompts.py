#!/usr/bin/env python
# coding: utf-8

# In[11]:

import copy
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
from scipy.special import log_softmax
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer

from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel

from utils.data import load_data

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_prompt_and_dev(k, prompt_seed, task_train_data, task_dev_data, sampling_weights, prompt_indices,
                       only_top_n=None, subsample_dev=None, from_one_of_wrong_tasks=None, random_prompt_labels=False):
    assert sampling_weights is None or prompt_indices is None
    random.seed(prompt_seed)
    np.random.seed(prompt_seed)
    if from_one_of_wrong_tasks is not None:
        wrong_task = random.sample(from_one_of_wrong_tasks, 1)[0]
        task_train_data = [dp for dp in task_train_data if dp['task'] == wrong_task]
    if subsample_dev is not None:
        curr_dev_data = random.sample(task_dev_data, subsample_dev)
    else:
        curr_dev_data = task_dev_data
    task_train_data = [{'index': i, 'instance': x} for i, x in enumerate(task_train_data)]
    if sampling_weights is not None:
        if only_top_n is not None:
            # sample k uniformly from the top n
            if only_top_n == 0:
                # if n is 0, set n = k
                only_top_n = k
            top_n_train_data = sorted(list(zip(task_train_data, sampling_weights)), key=lambda x: -x[1])[:only_top_n]
            # top_n_train_data = sorted(list(zip(task_train_data, sampling_weights)), key=lambda x: x[1])[:only_top_n]
            top_n_train_data, top_k_sampling_weights = zip(*top_n_train_data)
            curr_train_data = np.random.choice(top_n_train_data, size=k, replace=False).tolist()
        else:
            # sample k examples from the training data weighted by the weights given
            curr_train_data = np.random.choice(task_train_data, size=k, replace=False, p=sampling_weights).tolist()
        # to make sure example order is not impacted by weight
        curr_train_data = random.sample(curr_train_data, k)
    elif prompt_indices is not None:
        curr_train_data = [task_train_data[index] for index in prompt_indices]
    else:
        # sample k randomly from the train data
        curr_train_data = random.sample(task_train_data, k)
    train_indices = [x['index'] for x in curr_train_data]
    curr_train_data = [x['instance'] for x in curr_train_data]
    if random_prompt_labels:
        new_curr_train_data = []
        for dp in curr_train_data:
            dp = copy.deepcopy(dp)
            dp['output'] = random.sample(dp['options'], 1)[0]
            new_curr_train_data.append(dp)
        curr_train_data = new_curr_train_data
    return train_indices, curr_train_data, curr_dev_data


def get_performance(metaicl_model, metaicl_data, logger, seed, task, k, gpt2, checkpoint, prompt_seed, is_classification,
                    train_indices, curr_train_data,
                    curr_dev_data, args, train_data_split, add_newlines=True,
                    finetune_instead=False, option_normalization=True):
    # assert len(curr_dev_data)>0
    assert not args.use_demonstrations or len(curr_train_data)==k, \
            (args.use_demonstrations, len(curr_train_data), k)

    # config_file = "config/tasks/{}.json".format(task)
    # assert os.path.exists(config_file), config_file
    # with open(config_file, "r") as f:
    #     config = json.load(f)
    # is_classification = config["task_type"]=="classification"
    # if is_classification:
    #     options = curr_dev_data[0]["options"]
    #     assert np.all([d["options"]==options for d in curr_dev_data])

    loss, normalized_loss, acc, f1 = run(
        logger, task, metaicl_data, metaicl_model,
        curr_train_data, curr_dev_data, seed, checkpoint, is_classification,
        add_newlines, args,
        finetune_instead=finetune_instead, option_normalization=option_normalization)

    return {
        'k': args.k,
        'task': task,
        'prompt_seed': prompt_seed,
        'split': train_data_split,
        'train_slice_args': args.train_slice_args,
        'train_indices': str(train_indices),
        'train_samples': curr_train_data,
        'dev_slice_args': args.dev_slice_args,
        'subsample_dev': args.subsample_dev,
        'checkpoint': checkpoint,
        'gpt2': gpt2,
        'loss': loss,
        'normalized_loss': normalized_loss,
        'acc': acc,
        'f1': f1,
        'is_classification': is_classification,
    }


def get_out_name(out_dir, task, split_name, method, add_newlines, seed, args):
    return os.path.join(
        out_dir, "{}-{}-{}{}{}{}{}".format(
            task,
            split_name,
            method,
            "-k={}".format(args.k) if args.use_demonstrations else "",
            "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
            "" if add_newlines else "-no-newlines",
            "-randomEnglish" if args.use_random_english_words else ""))


original_state_dict = None
def run(logger, task, metaicl_data, metaicl_model, train_data, dev_data, seed,
        checkpoint, is_classification, add_newlines, args, finetune_instead=False,
        option_normalization=True):
    if finetune_instead:
        global original_state_dict
        metaicl_data.tensorize([], train_data, add_newlines=add_newlines)
        if metaicl_model.is_none():
            metaicl_model.load(checkpoint, gpt2=args.gpt2)
            assert not os.path.exists('temp_model_path.pt')
            original_state_dict = metaicl_model.model.state_dict()
            metaicl_model.to_device()
        else:
            metaicl_model.model.load_state_dict(original_state_dict)
        metaicl_model.setup_optimizer(args.optimization, args.num_training_steps, args.lr,
                                      args.weight_decay, args.warmup_steps)
        # metaicl_model.parallel()
        metaicl_model.train()
        metaicl_model.do_train(metaicl_data, args.batch_size, args.num_training_steps, 1000000000000000000, 10)
        metaicl_data.tensorize(train_data, dev_data, add_newlines=add_newlines)
        metaicl_model.eval()
    else:
        metaicl_data.tensorize(train_data, dev_data, add_newlines=add_newlines)
        metaicl_data.print_tensorized_example()
        if metaicl_model.is_none():
            metaicl_model.load(checkpoint, gpt2=args.gpt2)
            metaicl_model.cuda()
            # metaicl_model.parallel()
            metaicl_model.eval()

    if not option_normalization:
        answer_indices = [dp['indices'][dp['answer'][0]][0] for dp in metaicl_data.metadata]
        for k, v in metaicl_data.tensorized_inputs.items():
            if k == 'labels':
                raise NotImplementedError
            metaicl_data.tensorized_inputs[k] = v[torch.tensor(answer_indices)]
        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size, verbose=True)
        loss = losses.mean()
        normalized_loss = None
        acc = None
        f1 = None
    else:
        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size, verbose=True)
        # predictions = metaicl_model.do_predict(metaicl_data, losses=losses)
        losses = np.array(losses)
        assert len(losses)==len(metaicl_data)
        predictions = []
        answer_losses = []
        normalized_losses = []
        for idx, dp in enumerate(metaicl_data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())

            answer_idx = dp['answer'][0]
            answer_losses.append(curr_label_losses[answer_idx])
            normalized_losses.append(-log_softmax(-np.array(curr_label_losses))[answer_idx])
        loss = np.array(answer_losses).mean()
        normalized_loss = np.array(normalized_losses).mean()
        groundtruths = [dp["output"] for dp in dev_data]
        acc = metaicl_data.evaluate(predictions, groundtruths, False)
        f1 = metaicl_data.evaluate(predictions, groundtruths, True)
    return loss, normalized_loss, acc, f1


def get_random_preds_performance(task, metaicl_data, prompt_seed, is_classification, dev_data, split, args):
    groundtruths = [dp["output"] for dp in dev_data]
    predictions = [random.sample(dp["options"], 1)[0] for dp in dev_data]
    metaicl_data.metadata = [None] * len(dev_data)
    acc = metaicl_data.evaluate(predictions, groundtruths, False)
    f1 = metaicl_data.evaluate(predictions, groundtruths, True)
    return {
        'task': task,
        'prompt_seed': prompt_seed,
        'split': split,
        'dev_slice_args': args.dev_slice_args,
        'subsample_dev': args.subsample_dev,
        'acc': acc,
        'f1': f1,
        'is_classification': is_classification,
    }


def main(args, logger):

    from tqdm import tqdm
    errors = []

    for gpt2, chckpnt in zip(args.gpt2s.split(','), args.checkpoints.split(',')):
        args.gpt2 = gpt2
        args.checkpoint = chckpnt
        if args.gpt2.startswith("gpt2"):
            tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        add_newlines = True

        ### checkpoint ...
        if not args.do_zeroshot:
            if args.checkpoint is not None:
                checkpoint = args.checkpoint
                assert args.global_step is None
            else:
                assert args.global_step is not None
                checkpoint = os.path.join(args.out_dir, "model-{}.pt".format(args.global_step))
            # assert os.path.exists(checkpoint)
        else:
            add_newlines = not args.gpt2.startswith("gpt2")
            if args.gpt2=="gpt-j-6B":
                # we are using the HF veresion where GPT-J-6B checkpoint is not officially registered
                # so need to download the model checkpoint and specify checkpoint
                # assert args.checkpoint is not None and os.path.exists(args.checkpoint)
                args.gpt2 = args.checkpoint
            checkpoint = None
        metaicl_model = MetaICLModel(logger, args.out_dir)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        # setup hyperparams for data

        max_length_per_example = 256
        max_length = min(max_length_per_example * (args.k + 1), 1024)

        logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
            args.test_batch_size, max_length, max_length_per_example))

        metaicl_data = MetaICLData(logger, tokenizer, args.method, True, 32,
                                   max_length, max_length_per_example, tensorize_dir=args.tensorize_dir,
                                   # n_gpu=,
                                   local_rank=args.local_rank, do_tensorize=True, n_process=1)

        # In[ ]:

        seeds = args.seed.split(",")

        for seed in seeds:
            ### data ...
            train_data = load_data(args.task, "train", 1000, seed=seed, config_split=config_split,
                                   datasets=None)
            dev_data = load_data(args.task, args.split, 1000, seed=seed, config_split=config_split,
                                 datasets=None, is_null=args.is_null)

            if args.use_random_english_words:
                from english_words import english_words_set

                english_words_set = sorted(english_words_set)
                np.random.seed(int(seed))

            train_counter = Counter()
            dev_counter = Counter()
            for dp in train_data:
                train_counter[dp["task"]] += 1
            for dp in dev_data:
                dev_counter[dp["task"]] += 1
            for k, v in train_counter.items():
                logger.info("[Train] %s\t%d" % (k, v))
            for k, v in dev_counter.items():
                logger.info("[Dev] %s\t%d" % (k, v))

            logger.info("%s on %s (%d train, %d dev)" % (args.method, args.task, len(train_counter), len(dev_counter)))

            for test_task in (list(dev_counter) if args.dataset is None else args.dataset.split(',')):
                df = pd.DataFrame([], columns=[
                    'k', 'task', 'prompt_seed', 'split', 'train_slice_args', 'train_indices', 'train_samples',
                    'dev_slice_args', 'subsample_dev', 'checkpoint', 'gpt2', 'loss', 'normalized_loss', 'acc', 'f1'])
                all_task_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
                if args.prompt_with_random_examples_not_from_task or args.prompt_with_random_wrong_task:
                    all_task_train_data = [dp for dp in train_data if dp["task"]!=test_task]
                else:
                    all_task_train_data = [dp for dp in train_data if dp["task"]==test_task]
                for (split, train_proportion_slice_tuple, dev_proportion_slice_tuple, prompt_seeds, subsample_dev,
                     postfix) in args.splits:
                    args.train_slice_args = [int(x * len(all_task_train_data)) for x in train_proportion_slice_tuple]
                    args.dev_slice_args = [int(x * len(all_task_dev_data)) for x in dev_proportion_slice_tuple]
                    task_train_data = all_task_train_data[slice(*args.train_slice_args)]
                    task_dev_data = all_task_dev_data[slice(*args.dev_slice_args)]
                    args.total_data_size = len(task_train_data)
                    args.prompt_seeds = prompt_seeds
                    args.subsample_dev = subsample_dev
                    if os.path.exists(os.path.join(args.out_dir, 'results_%s%s.csv' % (test_task, postfix))):
                        df = pd.read_csv(os.path.join(args.out_dir, 'results_%s%s.csv' % (test_task, postfix)))

                    if args.sampling_weights_dir is not None:
                        assert not (args.prompt_with_random_examples_not_from_task or
                                    args.prompt_with_random_wrong_task or
                                    args.prompt_weights_dir is None)
                        # with open(os.path.join(args.sampling_weights_dir, test_task + '.pkl'), 'rb') as f:
                        task_weights = np.load(os.path.join(args.sampling_weights_dir, test_task + '.npy'))
                        task_weights = task_weights / task_weights.sum(keepdims=True)
                    else:
                        task_weights = None
                    if args.prompt_weights_dir is not None:
                        with open(os.path.join(args.prompt_weights_dir, test_task + '.pkl'), 'rb') as f:
                            task_prompt_weights = pkl.load(f)
                    else:
                        task_prompt_weights = None

                    if len(task_dev_data) == 0:
                        continue
                    # assert len(task_dev_data) > 0
                    # assert not args.use_demonstrations or len(task_train_data) == args.total_data_size or args.prompt_with_random_tasks, \
                    #     (args.use_demonstrations, len(task_train_data), args.total_data_size, args.prompt_with_random_tasks)

                    config_file = "config/tasks/{}.json".format(test_task)
                    assert os.path.exists(config_file), config_file
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    is_classification = config["task_type"] == "classification"
                    if is_classification:
                        options = task_dev_data[0]["options"]
                        assert np.all([d["options"] == options for d in task_dev_data])
                    if args.use_random_english_words:
                        # create a mapping
                        options = task_dev_data[0]["options"]
                        mapping = {option: np.random.choice(english_words_set) for option in options}
                        new_options = list(mapping.values())
                        for dp_idx, dp in enumerate(task_train_data):
                            assert dp["output"] in options, (dp, options)
                            task_train_data[dp_idx]["output"] = mapping[dp["output"]]
                            task_train_data[dp_idx]["options"] = new_options
                        for dp_idx, dp in enumerate(task_dev_data):
                            assert dp["output"] in options, (dp, options)
                            task_dev_data[dp_idx]["output"] = mapping[dp["output"]]
                            task_dev_data[dp_idx]["options"] = new_options

                    if task_prompt_weights is None:
                        def k_to_seeds(k):
                            # if args.subsample_dev is None:
                            #     if k == 0:
                            #         return [0]
                            #     if args.sampling_weights_dir is not None and (k == args.top_n or args.top_n == 0):
                            #         return [0]
                            return args.prompt_seeds
                        iterable = [
                            (curr_k, prompt_seed) for curr_k in args.ks
                            for prompt_seed in k_to_seeds(curr_k)]
                        print(iterable)
                        iterable = tqdm(iterable, total=len(iterable))
                    else:
                        iterable = [prompt_info for prompt_info in task_prompt_weights]
                    for x in iterable:
                        if task_prompt_weights is None:
                            curr_k, prompt_seed = x
                            prompt_indices = None
                        else:
                            prompt_indices = x['prompt_indices']
                            prompt_seed = x['prompt_seed']
                            curr_k = len(prompt_indices)
                        args.k = curr_k
                        metaicl_data.k = curr_k if not args.finetune else 0
                        args.use_demonstrations = metaicl_data.k != 0
                        metaicl_data.use_demonstrations = metaicl_data.k != 0
                        train_indices, curr_train_data, curr_dev_data = get_prompt_and_dev(
                            args.k, prompt_seed, task_train_data, task_dev_data,
                            task_weights, prompt_indices, only_top_n=args.top_n, subsample_dev=args.subsample_dev,
                            from_one_of_wrong_tasks=[t for t in sorted(list(dev_counter)) if t != test_task] \
                                if args.prompt_with_random_wrong_task else None,
                            random_prompt_labels=args.random_prompt_labels
                        )
                        if args.random_preds:
                            rows = df[
                                (df.task == test_task) &
                                (df.prompt_seed == prompt_seed) &
                                (df.dev_slice_args == str(args.dev_slice_args)) &
                                ((df.subsample_dev == subsample_dev)
                                 if subsample_dev is not None else (df.index == df.index)) &
                                (df['split'] == split)]
                            if len(rows) > 0:
                                print('found', x)
                                print(rows)
                                continue
                            result = get_random_preds_performance(
                                test_task, metaicl_data, prompt_seed, is_classification, curr_dev_data, split, args)
                        else:
                            rows = df[
                                (df.k == curr_k) &
                                (df.task == test_task) &
                                (df.prompt_seed == prompt_seed) &
                                (df.train_slice_args == str(args.train_slice_args)) &
                                (df.dev_slice_args == str(args.dev_slice_args)) &
                                ((df.subsample_dev == subsample_dev)
                                 if subsample_dev is not None else (df.index == df.index)) &
                                # (df.train_indices == str(train_indices)) &
                                (df.checkpoint == checkpoint) &
                                (df['split'] == split)]
                            if len(rows) > 0:
                                if task_prompt_weights is None:
                                    print('found', x)
                                    print(rows)
                                continue
                            result = get_performance(
                                metaicl_model, metaicl_data, logger, seed, test_task, args.k, args.gpt2,
                                checkpoint, prompt_seed, is_classification, train_indices, curr_train_data,
                                curr_dev_data, args, split, finetune_instead=args.finetune
                            )
                        if isinstance(result, dict):
                            df = pd.concat([df, pd.DataFrame([result])])
                            df.to_csv(os.path.join(args.out_dir, 'results_%s%s.csv' % (test_task, postfix)),
                                      index=False)
                        else:
                            errors.append(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, help="output directory", required=True)
    parser.add_argument("-s", "--splits", nargs='+',
                        help="Something like: \"('train', (0, .5), (0, .06), range(125), None, '_1')\"", required=True)
    parser.add_argument("--test_batch_size", type=int, default=16, help="output directory")
    parser.add_argument("--gpt2s", type=str, default='gpt2-large')
    parser.add_argument("--checkpoints", type=str, default='checkpoints/metaicl/hr_to_lr/model.pt')
    parser.add_argument("--method", type=str, default='channel')
    parser.add_argument("--task", type=str, default='hr_to_lr')
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--prompt_with_random_examples_not_from_task", default=False, action='store_true')
    parser.add_argument("--prompt_with_random_wrong_task", default=False, action='store_true')
    parser.add_argument("--random_preds", default=False, action='store_true')
    parser.add_argument("--random_prompt_labels", default=False, action='store_true')
    parser.add_argument("--ks", type=str, default='16')
    # args.gpt2s = 'gpt2-large'
    # args.checkpoints = 'checkpoints/metaicl/hr_to_lr/model.pt'
    # args.gpt2s = 'gpt2-large'
    # args.checkpoints = 'gpt2-large'
    # args.gpt2s = 'gpt-j-6B'
    # args.checkpoints = 'gpt-j-6B'
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()


    assert not (args.prompt_with_random_examples_not_from_task and args.prompt_with_random_wrong_task)
    # args = Namespace()
    args.global_step = None
    args.do_zeroshot = False
    args.k = 32
    args.splits = [eval(split) for split in args.splits]
    config_split = "test"
    # config_split = "train"
    args.split = 'dev'
    args.is_null = False
    args.out_dir = os.path.join('/scratch/mcinerney.de/metaicl', args.out_dir)
    args.seed = '100'
    args.use_random_english_words = False
    args.use_calibration = False
    args.ks = [int(x) for x in args.ks.split(',')]
    args.sampling_weights_dir = None
    # args.sampling_weights_dir = 'regcoef/checkpoints-metaicl-hr_to_lr-model.pt'
    # args.sampling_weights_dir = 'negregcoef/checkpoints-metaicl-hr_to_lr-model.pt'
    # args.sampling_weights_dir = 'regcoef/gpt-j-6B'
    # args.sampling_weights_dir = 'negregcoef/gpt-j-6B'
    # args.sampling_weights_dir = 'results/results_gpt2/uncertainty_sampling'
    args.top_n = 32
    # args.top_n = 0
    # args.prompt_weights_dir = 'results/results_gpt2/pero'
    args.prompt_weights_dir = None
    args.finetune = False
    args.lr = 3e-5
    args.warmup_steps = 0
    args.batch_size = 16
    args.num_training_steps = 100
    args.weight_decay = 0.0
    args.optimization = "adamw"
    args.tensorize_dir = 'tensorize_dir'


    # config_split = "unseen_domain_test" if args.unseen_domain_only else "test"
    handlers = [logging.StreamHandler()]
    # if args.log_file is not None:
    #     handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(args, logger)