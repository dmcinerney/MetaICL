#!/usr/bin/env python
# coding: utf-8

# In[11]:


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

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_performance(task, task_train_data, task_dev_data, k, trim_dev_data, gpt2, checkpoint, prompt_seed,
                    sampling_weights, is_classification, add_newlines=True, save_predictions=True, only_top_n=None):
    random.seed(prompt_seed)
    np.random.seed(prompt_seed)
    #                 curr_dev_data = random.sample(task_train_data, args.trim_dev_data)
    curr_dev_data = task_dev_data[:trim_dev_data] if trim_dev_data is not None else task_dev_data
    if sampling_weights is not None:
        if only_top_n is not None:
            # sample k uniformly from the top n
            if only_top_n == 0:
                # if n is 0, set n = k
                only_top_n = k
            top_n_train_data = sorted(list(zip(task_train_data, sampling_weights)), key=lambda x: x[1])[:only_top_n]
            top_n_train_data, top_k_sampling_weights = zip(*top_n_train_data)
            curr_train_data = np.random.choice(top_n_train_data, size=k, replace=False).tolist()
        else:
            # sample k examples from the training data weighted by the weights given
            curr_train_data = np.random.choice(task_train_data, size=k, replace=False, p=sampling_weights).tolist()
        # to make sure example order is not impacted by weight
        curr_train_data = random.sample(curr_train_data, k)
    else:
        # sample k randomly from the train data
        curr_train_data = random.sample(task_train_data, k)
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

    result = run(logger, task, metaicl_data, metaicl_model,
                 curr_train_data, curr_dev_data, seed, checkpoint, is_classification, add_newlines, args,
                 save_predictions=save_predictions)

    if result is None:
        errors.append("%s/%s" % (task, seed))
    else:
        return {
            'k': args.k,
            'task': task,
            'prompt_seed': prompt_seed,
            'train_samples': curr_train_data,
            'result': result,
            'checkpoint': checkpoint,
            'gpt2': gpt2,
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


def run(logger, task, metaicl_data, metaicl_model, train_data, dev_data, seed,
        checkpoint, is_classification, add_newlines, args, save_predictions=True):
#     import pdb; pdb.set_trace()

    if args.do_zeroshot:
        split_name = args.split
        if args.is_null:
            split_name += "-null"
        cache_path = get_out_name(args.out_dir, task, split_name, metaicl_data.method, add_newlines, seed, args)+ ".pkl"
    else:
        assert add_newlines
        cache_path = get_out_name(args.out_dir, task, args.split, metaicl_data.method, False, seed, args)+ ".pkl"

    metaicl_data.tensorize(train_data, dev_data, add_newlines=add_newlines)
    metaicl_data.print_tensorized_example()
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")

#     if os.path.exists(prediction_path):
#         return 0

#     if os.path.exists(cache_path):
#         with open(cache_path, "rb") as f:
#             losses = pkl.load(f)
    if False:
        pass
    else:
        if metaicl_model.is_none():
            metaicl_model.load(checkpoint, gpt2=args.gpt2)
            metaicl_model.cuda()
            metaicl_model.eval()

        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size, verbose=True)
        # with open(cache_path, "wb") as f:
        #     pkl.dump(losses, f)

    assert len(losses)==len(metaicl_data)

    if args.is_null:
        return None

    # if args.use_calibration:
    #     assert args.do_zeroshot
    #     bias_path = cache_path.replace("/"+task+"-"+args.split, "/"+task+"-"+args.split+"-null")
    #     assert os.path.exists(bias_path), bias_path
    #     with open(bias_path, "rb") as f:
    #         bias_losses = pkl.load(f)
    #
    #     losses = np.array(losses)
    #     bias_losses = np.array(bias_losses)
    #     assert losses.shape == bias_losses.shape
    #     losses -= bias_losses

    predictions = metaicl_model.do_predict(metaicl_data, losses=losses)
    groundtruths = [dp["output"] for dp in dev_data]
    perf = metaicl_data.evaluate(predictions, groundtruths, is_classification)
    logger.info("Accuracy=%s" % perf)

    if save_predictions:
        with open(prediction_path, "w") as f:
            for prediction in predictions:
                f.write(prediction)
                f.write("\n")

    return perf


if __name__ == '__main__':
    args = Namespace()
    args.gpt2 = 'gpt2-large'
    args.do_zeroshot = False
    args.checkpoint = 'checkpoints/metaicl/hr_to_lr/model.pt'
    args.global_step = None
    args.use_demonstrations = True
    args.do_zeroshot = False
    args.k = 16
    args.total_data_size = 200
    args.out_dir = None
    args.test_batch_size = 8
    # args.test_batch_size = 16
    args.method = 'direct'
    args.task = 'custom'
    args.unseen_domain_only = False
    args.dataset = None
    args.split = 'dev'
    args.is_null = False
    # args.out_dir = 'results/results_gptj'
    # args.out_dir = 'results/results_gpt2'
    args.out_dir = 'results/results_gpt2finetuned_uncertainty_sampling_top_n32'
    # args.out_dir = 'results/results_gpt2_prompt_with_random_tasks'
    # args.out_dir = 'results/results_gpt2finetuned_prompt_with_random_tasks'
    # args.out_dir = 'results/results_gpt2_uncertainty_sampling'
    # args.out_dir = 'results/results_gpt2_uncertainty_sampling_top_n32'
    args.seed = '100'
    args.use_random_english_words = False
    args.use_calibration = False
    args.num_prompt_samples = 32
    args.ks = [0, 1, 2, 4, 8, 16, 32]
    args.trim_dev_data = None
    args.gpt2s = 'gpt2-large'
    args.checkpoints = 'checkpoints/metaicl/hr_to_lr/model.pt'
    # args.gpt2s = 'gpt2-large'
    # args.checkpoints = 'gpt2-large'
    # args.gpt2s = 'gpt-j-6B'
    # args.checkpoints = 'gpt-j-6B'
    args.prompt_with_random_tasks = False
    args.sampling_weights_dir = 'results/results_gpt2/uncertainty_sampling'
    args.sampling_weights_dir = None
    args.top_n = 32
    # args.top_n = 0


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


    # In[5]:


    from tqdm import tqdm
    df = pd.DataFrame([], columns=['k', 'task', 'prompt_seed',  'train_samples', 'result', 'gpt2', 'checkpoint'])
    if os.path.exists(os.path.join(args.out_dir, 'results.csv')):
        df = pd.read_csv(os.path.join(args.out_dir, 'results.csv'))
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
            dev_data = load_data(args.task, args.split, args.total_data_size, seed=seed, config_split=config_split,
                                 datasets=None if args.dataset is None else args.dataset.split(","), is_null=args.is_null)

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

            for test_task in list(dev_counter):
                task_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
                if args.prompt_with_random_tasks:
                    task_train_data = [dp for dp in train_data if dp["task"]!=test_task]
                else:
                    task_train_data = [dp for dp in train_data if dp["task"]==test_task]

                if args.sampling_weights_dir is not None:
                    assert not args.prompt_with_random_tasks
                    # with open(os.path.join(args.sampling_weights_dir, test_task + '.pkl'), 'rb') as f:
                    task_weights = np.load(os.path.join(args.sampling_weights_dir, test_task + '.npy'))
                    task_weights = task_weights / task_weights.sum(keepdims=True)
                else:
                    task_weights = None

                assert len(task_dev_data) > 0
                assert not args.use_demonstrations or len(task_train_data) == args.total_data_size or args.prompt_with_random_tasks, \
                    (args.use_demonstrations, len(task_train_data), args.total_data_size, args.prompt_with_random_tasks)

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

                for curr_k in args.ks:
                    print('for k = %i' % curr_k)
                    args.k = curr_k
                    metaicl_data.k = curr_k
                    for prompt_seed in tqdm(range(args.num_prompt_samples), total=args.num_prompt_samples) if curr_k > 0 else [0]:
                        rows = df[
                            (df.task == test_task) &
                            (df.checkpoint == checkpoint) &
                            (df.k == args.k) &
                            (df.prompt_seed == prompt_seed)]
                        if len(rows) > 0:
                            continue
                        result = get_performance(
                            test_task, task_train_data, task_dev_data, args.k, args.trim_dev_data, args.gpt2,
                            checkpoint, prompt_seed, task_weights, is_classification, save_predictions=False,
                            only_top_n=args.top_n
                        )
                        if isinstance(result, dict):
                            df = pd.concat([df, pd.DataFrame([result])])
                            df.to_csv(os.path.join(args.out_dir, 'results.csv'), index=False)
                        else:
                            errors.append(result)

    # logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.task, len(results) // len(seeds), 100*np.mean(results)))

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


    # In[ ]:


    # df = pd.DataFrame(results)
    # df.to_csv(os.path.join(args.out_dir, 'results.csv'), index=False)
    # df
    #
    #
    # # In[6]:
    #
    #
    # df = pd.read_csv(os.path.join(args.out_dir, 'results.csv'))
    # df
    #
    #
    # # In[7]:
    #
    #
    # import seaborn as sns
    # sns.violinplot(data=df, x='k', y='result', hue=['task', 'checkpoint'])
    # plt.savefig(os.path.join(args.out_dir, 'accuracy_dists.pdf'))


#     # In[8]:


#     import pprint
#     for test_task in args.dataset.split(','):
#         print('\n\nTask: ' + test_task)
#         print('Worst\n')
#         pprint.pprint(eval(df[(df.k == 2) & (df.task == test_task)].sort_values('result').iloc[0].train_samples))
#         print('\nBest\n')
#         pprint.pprint(eval(df[(df.k == 2) & (df.task == test_task)].sort_values('result').iloc[-1].train_samples))


#     # In[9]:


#     seed = 100
#     test_task = 'piqa'
#     train_data = load_data(
#         args.task, "train", 200, seed=seed, config_split=config_split,
#         datasets=[test_task])
#     dev_data = load_data(
#         args.task, args.split, 200, seed=seed, config_split=config_split,
#         datasets=[test_task], is_null=args.is_null)


#     # In[12]:


#     import copy
#     args_temp = copy.deepcopy(args)
#     curr_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
#     for k in args.ks:
#         args_temp.k = k
#         metaicl_data.k = k
#         df_k_sorted = df[(df.k == k) & (df.task == test_task)].sort_values('result')
#         curr_train_data = eval(df_k_sorted.iloc[0].train_samples)
#         assert len(curr_dev_data)>0
#         assert not args_temp.use_demonstrations or len(curr_train_data)==args_temp.k, \
#                 (args_temp.use_demonstrations, len(curr_train_data), args_temp.k)

#         config_file = "config/tasks/{}.json".format(test_task)
#         assert os.path.exists(config_file), config_file
#         with open(config_file, "r") as f:
#             config = json.load(f)
#         is_classification = config["task_type"]=="classification"
#         if is_classification:
#             options = curr_dev_data[0]["options"]
#             assert np.all([d["options"]==options for d in curr_dev_data])
#         args_temp.out_dir = os.path.join(args.out_dir, 'worst')
#         if not os.path.exists(args_temp.out_dir):
#             os.mkdir(args_temp.out_dir)
#         result = run(
#             logger, test_task, metaicl_data, metaicl_model,
#             curr_train_data, curr_dev_data, seed, checkpoint, is_classification, add_newlines, args_temp)
#         args_temp.out_dir = os.path.join(args.out_dir, 'best')
#         if not os.path.exists(args_temp.out_dir):
#             os.mkdir(args_temp.out_dir)
#         curr_train_data = eval(df_k_sorted.iloc[-1].train_samples)
#         result = run(
#             logger, test_task, metaicl_data, metaicl_model,
#             curr_train_data, curr_dev_data, seed, checkpoint, is_classification, add_newlines, args_temp)


#     # In[23]:


#     import difflib
#     from sentence_transformers import SentenceTransformer
#     from sklearn.manifold import TSNE
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     args_temp = copy.deepcopy(args)
#     for k in args.ks:
#         args_temp.k = k
#         metaicl_data.k = k
#         print(k)
#         df_k_sorted = df[(df.k == k) & (df.task == test_task)].sort_values('result')
#         out_path = get_out_name(
#             os.path.join(args.out_dir, 'worst'), test_task, args.split, metaicl_data.method, False, seed, args_temp) + '.txt'
#         with open(out_path, 'r') as f:
#             worst_answers = f.readlines()
#         out_path = get_out_name(
#             os.path.join(args.out_dir, 'best'), test_task, args.split, metaicl_data.method, False, seed, args_temp) + '.txt'
#         with open(out_path, 'r') as f:
#             best_answers = f.readlines()
#     #     for line in difflib.unified_diff(worst_answers, best_answers, fromfile='worst', tofile='best', lineterm=''):
#     #         print(line)
#         dev_embs = model.encode([dp['input'] for dp in dev_data if dp["task"]==test_task])
#         train_embs_best = model.encode([dp['input'] for dp in eval(df_k_sorted.iloc[-1].train_samples)])
#         train_embs_worst = model.encode([dp['input'] for dp in eval(df_k_sorted.iloc[0].train_samples)])
#         if len(train_embs_best) == 0:
#             train_embs_best = train_embs_best.reshape(0, dev_embs.shape[1])
#             train_embs_worst = train_embs_worst.reshape(0, dev_embs.shape[1])
#         tsne = TSNE()
#         transformed = tsne.fit_transform(np.concatenate([dev_embs, train_embs_best, train_embs_worst]))
#         split1, split2 = len(dev_embs), len(dev_embs) + len(train_embs_best)
#         dev_embs_transformed, train_embs_best_transformed, train_embs_worst_transformed = \
#             transformed[:split1], transformed[split1:split2], transformed[split2:]
#         embeddings = []
#         for (x, y), bestpred, worstpred, dp in zip(dev_embs_transformed, best_answers, worst_answers, [
#             dp for dp in dev_data if dp["task"]==test_task]):
#             bestpred, worstpred = bestpred.strip(), worstpred.strip()
#             embeddings.append({
#                 'x': x,
#                 'y': y,
#                 '': 'dev-same' if bestpred == worstpred else 'dev-corrected' if bestpred == dp['output'] else 'dev-corrupted',
#                 'color': 'blue' if bestpred == worstpred else 'green' if bestpred == dp['output'] else 'red',
#                 'size': 1 if bestpred == worstpred else 2,
#             })

#         for x, y in train_embs_best_transformed:
#             embeddings.append({
#                 'x': x,
#                 'y': y,
#                 '': 'train-best',
#                 'color': 'red',
#                 'size': 3,
#             })
#         for x, y in train_embs_worst_transformed:
#             embeddings.append({
#                 'x': x,
#                 'y': y,
#                 '': 'train-worst',
#                 'color': 'green',
#                 'size': 3,
#             })
#         embeddings = pd.DataFrame(embeddings)
#         # p = sns.scatterplot(data=embeddings, x='x', y='y', style='', hue='',
#         #                     markers={'dev-corrected': 'o', 'dev-corrupted': 'o', 'dev-same': '.', 
#         #                              "train-worst": "s", "train-best": "X"})
#         sns.scatterplot(data=embeddings[embeddings[''] == 'dev-same'], x='x', y='y', color='blue', marker='.')
#         sns.scatterplot(data=embeddings[embeddings[''] == 'dev-corrected'], x='x', y='y', color='green', marker='o')
#         sns.scatterplot(data=embeddings[embeddings[''] == 'dev-corrupted'], x='x', y='y', color='red', marker='o')
#         sns.scatterplot(data=embeddings[embeddings[''] == 'train-best'], x='x', y='y', color='green', marker='+', s=200)
#         sns.scatterplot(data=embeddings[embeddings[''] == 'train-worst'], x='x', y='y', color='brown', marker='x', s=100)
#         # p.axis([-40, 25, -25, 25])
#         plt.savefig(os.path.join(args.out_dir, 'tsne_embedded_inputs_k=%i.pdf' % k))
#         plt.show()


#     # In[ ]:
