# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import string
import numpy as np
import torch

def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data = []
    for dataset in datasets:
        # data_path = os.path.join("data", dataset,
        #                          "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        data_path = os.path.join('/scratch/mcinerney.de/metaicl', "data", dataset,
                                 "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)
    return data



import torch
import torch.nn.functional as F


def get_max_dims(tensors):
    """
    Returns None if the tensors are all the same size and the maximum size in
    each dimension otherwise
    """
    if len(tensors) <= 0:
        return None
    dim = tensors[0].dim()
    max_size = [0]*dim
    different = False
    for tensor in tensors:
        if tensor.dim() != dim:
            raise Exception
        for i in range(dim):
            if not different:
                different = max_size[i] != tensor.size(i)
            max_size[i] = max(max_size[i], tensor.size(i))
    if different:
        return max_size
    else:
        return None


def pad_and_concat(tensors, max_size=None, auto=True):
    """
    Returns concatenated tensors with the added batch dimension being first
    """
    if auto:
        if max_size is not None:
            raise Exception("Must turn auto off to specify max size.")
        max_size = get_max_dims(tensors)
    concatenated_tensor = []
    for i,tensor in enumerate(tensors):
        if i == 0:
            dim = tensor.dim()
        elif tensor.dim() != dim:
            raise Exception("Number of dimensions does not match!")
        if max_size is not None:
            padding = []
            for i in range(dim-1, -1, -1):
                diff = max_size[i]-tensor.size(i)
                if diff < 0:
                    raise Exception(
                        "Tensor dim greater than specified max size!")
                padding.extend([0, diff])
            new_tensor = F.pad(tensor, tuple(padding))
        else:
            if i == 0:
                shape = tensor.shape
            elif tensor.shape != shape:
                raise Exception(
                    "When auto is turned off and max_size is None, "\
                    + "tensor shapes must match!")
            new_tensor = tensor
        concatenated_tensor.append(new_tensor.view(1, *new_tensor.size()))
    concatenated_tensor = torch.cat(concatenated_tensor, 0)
    return concatenated_tensor
