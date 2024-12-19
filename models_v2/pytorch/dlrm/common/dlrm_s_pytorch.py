# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import sys
import time


# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.profiler import profile, record_function, ProfilerActivity

from torch.utils import ThroughputBenchmark

# int8
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
# For distributed run
import extend_distributed as ext_dist

import os
import psutil

exc = getattr(builtins, "IOError", "FileNotFoundError")

first_iteration_for_train = True

buffer_num = 20
data_buffer = None
data_iter = None
train_ld = None

def freeze(model):
    return torch.jit._recursive.wrap_cpp_module(torch._C._freeze_module(model._c, preserveParameters=True))


def time_wrap():
    return time.time()


def dlrm_wrap(X, lS_o, lS_i, is_train=False):
    with record_function("DLRM forward"):
        return dlrm(X, lS_o, lS_i, is_train)


def loss_fn_wrap(Z, T):
    with record_function("DLRM loss compute"):
        return dlrm.loss_fn(Z, T)

# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None

def load_data(buffer_num, is_bf16):
    reset = False
    with torch.autograd.profiler.record_function('load_data'):
        for d in range(buffer_num):
            # (X, lS_i, T) = next(data_iter)
            try:
                Batch = next(data_iter)
            except:
                print("epoch ended, reset data_iter")
                reset = True
                data_iter._reset(train_ld)
                Batch = next(data_iter)
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(Batch)
            lS_i = lS_i.contiguous()
            X = X.bfloat16() if is_bf16 else X
            data_buffer[d] = (X, lS_o, lS_i, T, W, CBPP)
    return reset

class AlltoallOutputs():
    def __init__(self):
        self.data = None
        self.tail_data = None

    def get_data_base_on_input(self, input, shape):
        if self.data is None:
            self.data = input.new_empty(shape)
            return self.data
        if self.data.shape == torch.Size(shape):
            return self.data
        else:
            if self.tail_data is None:
                self.tail_data = input.new_empty(shape)
            return self.tail_data

class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)
            self.numel += (LL.weight.numel() + LL.bias.numel())

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, local_ln_emb_sparse, ln_emb_dense, np_init_emb_weight=False):
        total_numel = 0
        for n in ln:
            total_numel += m * n
        print("start to create emb, minimally need ~{} G memory ".format(total_numel * 4 / 1024 / 1024 / 1024))
        print("may actually use more memory depends on memory allocator like tcmalloc/jemalloc")
        print("current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        emb_l = nn.ModuleList()
        emb_dense = nn.ModuleList()
        emb_sparse = nn.ModuleList()
        #n_embs = ln.size if local_ln_emb is None else len(local_ln_emb)
        embs = range(len(ln))
        if local_ln_emb_sparse or ln_emb_dense:
            embs = local_ln_emb_sparse + ln_emb_dense
        for i in embs:
            n = ln[i]
            
            # TODO: the previous implementation uses sparse_grad = n >= self.sparse_dense_boundary and ext_dist.my_size > 1,
            # making sparse_grad a numpy.bool_ instead of a python bool, which is unsupported in TorchInductor.
            # Update the implementation to make sparse_grad a python bool.
            sparse_grad = True if n >= self.sparse_dense_boundary and ext_dist.my_size > 1 else False
            if np_init_emb_weight:
                W = np.random.uniform(
                        low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                    ).astype(np.float32)
                # W = np.zeros(shape=(n, m)).astype(np.float32)
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=sparse_grad, _weight=torch.from_numpy(W).requires_grad_())
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=sparse_grad)
            if ext_dist.my_size > 1 and n >= self.sparse_dense_boundary:
                emb_sparse.append(EE)
            elif ext_dist.my_size > 1:
                emb_dense.append(EE)
            else:
                emb_l.append(EE)
        print("create emb done, current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        self.numel += total_numel
        return emb_l, emb_dense, emb_sparse

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        weighted_pooling=None,
        loss_threshold=0.0,
        np_init_emb_weight=False,
        sparse_dense_boundary = 2048,
    ):
        print("np_init_emb_weight:" , np_init_emb_weight)
        super(DLRM_Net, self).__init__()
        self.numel = 0
        self.loss_threshold = loss_threshold

        n_emb = len(ln_emb)
        self.sparse_dense_boundary = sparse_dense_boundary
       
        self.ln_emb_dense = [i for i in range(n_emb) if ln_emb[i] < self.sparse_dense_boundary]
        self.ln_emb_sparse = [i for i in range(n_emb) if ln_emb[i] >= self.sparse_dense_boundary]
        
        #If running distributed, get local slice of embedding tables
        self.rank = -1
        if ext_dist.my_size > 1:
            self.rank = ext_dist.dist.get_rank()
            n_emb_sparse = len(self.ln_emb_sparse)
            self.n_local_emb_sparse, self.n_sparse_emb_per_rank = ext_dist.get_split_lengths(n_emb_sparse)
            self.local_ln_emb_sparse_slice = ext_dist.get_my_slice(n_emb_sparse)
            self.local_ln_emb_sparse = self.ln_emb_sparse[self.local_ln_emb_sparse_slice]
        self.l_emb_seeds = np.random.randint(low=0, high=100000, size=len(ln_emb))
        print("#############seed self.l_emb_seeds:", self.l_emb_seeds)
        self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
        self.top_l = self.create_mlp(ln_top, sigmoid_top)
        self.all2all_validation_outputs0 = AlltoallOutputs()
        self.all2all_validation_outputs1 = AlltoallOutputs()
        self.output_ind = AlltoallOutputs()
        self.validation_output_ind0 = AlltoallOutputs()
        self.validation_output_ind1 = AlltoallOutputs()
        if ext_dist.my_size > 1:
            self.emb_l, self.emb_dense, self.emb_sparse = self.create_emb(m_spa, ln_emb, self.local_ln_emb_sparse, self.ln_emb_dense,np_init_emb_weight)
        else:
            self.emb_l, _, _ = self.create_emb(m_spa, ln_emb, None, None, np_init_emb_weight)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.load_data_time = 0
                
    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
         # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, emb_l, lS_o, lS_i):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups
        is_ddp = isinstance(emb_l,  ext_dist.DDP)
        module_to_check = emb_l if not is_ddp else emb_l.module
        if args.ipex_interaction and isinstance(module_to_check, ipex.nn.modules.MergedEmbeddingBag):
            n_tables = module_to_check.n_tables
            indices = [lS_i[i] for i in range(n_tables)]
            offsets = [lS_o[i] for i in range(n_tables)]
            return emb_l(indices, offsets)

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
            )

            ly.append(V)

        return ly

    def interact_features(self, x, ly):
        if args.ipex_interaction:
            T = [x] + list(ly)
            R = ipex.nn.functional.interaction(*T)
        else:
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        return R

    def forward(self, dense_x, lS_o, lS_i, is_train=False):
        if ext_dist.my_size > 1:
            return self.distributed_forward(dense_x, lS_o, lS_i, is_train=is_train)
        else:
            return self.sequential_forward(dense_x, lS_o, lS_i)

    def distributed_forward_2iter_overlap(self, cur_data, next_data):
        if not cur_data[0]:
            return None, None

        dense_x0 = cur_data[0]['dense_x']
        T_test0 = cur_data[0]['T_test']
        lS_i0 = cur_data[0]['lS_i']
        lS_o0 = cur_data[0]['lS_o']
        batch_size0 = dense_x0.size()[0]

        lS_i_dense0 = [lS_i0[i] for i in self.ln_emb_dense]
        lS_i_sparse0 = [lS_i0[i] for i in self.ln_emb_sparse]

        if not cur_data[1]:
            lS_i_sparse0 = ext_dist.shuffle_data(lS_i_sparse0)
            g_i_sparse0 = [lS_i_sparse0[:, i * batch_size0:(i + 1) * batch_size0].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            offset = torch.arange(batch_size0 * ext_dist.my_size)
            g_o_sparse0 = [offset for i in range(self.n_local_emb_sparse)]

            # sparse embeddings
            with torch.autograd.profiler.record_function('Prof_sparse_emb_forward'):
                ly_sparse0 = self.apply_emb(dlrm.emb_sparse, g_o_sparse0, g_i_sparse0)

            with torch.autograd.profiler.record_function('Prof_alltoall_emb_forward'):
                a2a_req0 = ext_dist.alltoall(ly_sparse0, self.n_sparse_emb_per_rank)

            # dense embeddings
            with torch.autograd.profiler.record_function('Prof_dense_emb_forward'):
                ly_dense0 = self.apply_emb(batch_size0, lS_i_dense0, self.emb_dense)

            # bottom mlp
            # with torch.autograd.profiler.record_function('Prof_bot_mlp_forward'):
            x0 = self.apply_mlp(dense_x0, self.bot_l)
            with torch.autograd.profiler.record_function('Prof_alltoall_emb_wait_forward'):
                ly_sparse0 = a2a_req0.wait()

            # concat emb data for split sparse embs
            ly_sparse0_full = []
            if ext_dist.my_size > len(self.ln_emb_sparse):
                for i in range(len(self.ln_emb_sparse)):
                    ly_sparse0_split = torch.cat([ly_sparse0[j] for j in range(i, ext_dist.my_size, len(self.ln_emb_sparse))], 1)
                    ly_sparse0_full.append(ly_sparse0_split)
            else:
                ly_sparse0_full = list(ly_sparse0)

            ly0 = list(ly_dense0) + ly_sparse0_full

            with torch.autograd.profiler.record_function('Prof_interaction_forward'):
                z0 = self.interact_features(x0, ly0)
            # top mlp
            with torch.autograd.profiler.record_function('Prof_top_mlp_forward'):
                p0 = self.apply_mlp(z0, self.top_l)

            # clamp output if needed
            if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
                z0 = torch.clamp(
                    p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
                )
            else:
                z0 = p0
            z0 = z0.float()

            Z_test0 = ext_dist.all_gather(z0, None)
            T_test0 = ext_dist.all_gather(T_test0, None)

            return (Z_test0, T_test0), None

        # if not cur_data[1]:
        #     self.distributed_forward(cur_data[0])

        if cur_data[1]:
            dense_x1 = cur_data[1]['dense_x']
            T_test1 = cur_data[1]['T_test']
            lS_i1 = cur_data[1]['lS_i']
            lS_o1 = cur_data[1]['lS_o']
            batch_size1 = dense_x1.size()[0]

            lS_i_dense1 = [lS_i1[i] for i in self.ln_emb_dense]
            lS_i_sparse1 = [lS_i1[i] for i in self.ln_emb_sparse]

        if next_data[0]:
            next_dense_x0 = next_data[0]['dense_x']
            # next_T_test0 = next_data[0]['T_test']
            next_lS_i0 = next_data[0]['lS_i']
            next_lS_o0 = next_data[0]['lS_o']
            next_batch_size0 = next_dense_x0.size()[0]

            # next_lS_i_dense0 = [lS_i0[i] for i in self.ln_emb_dense]
            next_lS_i_sparse0 = [next_lS_i0[i] for i in self.ln_emb_sparse]

        if next_data[1]:
            next_dense_x1 = next_data[1]['dense_x']
            # next_T_test1 = next_data[1]['T_test']
            next_lS_i1 = next_data[1]['lS_i']
            next_lS_o1 = next_data[1]['lS_o']
            next_batch_size1 = next_dense_x1.size()[0]

            # next_lS_i_dense1 = [lS_i1[i] for i in self.ln_emb_dense]
            next_lS_i_sparse1 = [next_lS_i1[i] for i in self.ln_emb_sparse]

        if 'lS_i_sparse_res' not in cur_data[0]:
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                lS_i_sparse0 = torch.cat([lS_i_sparse0 for _ in range(num_split_grps) ])
            lS_i_sparse0 = ext_dist.shuffle_data(lS_i_sparse0)
            g_i_sparse0 = [lS_i_sparse0[:, i * batch_size0:(i + 1) * batch_size0].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
        else:
            g_i_sparse0 = cur_data[0]['lS_i_sparse_res']

        if cur_data[1] and 'lS_i_sparse_res' not in cur_data[1]:
            lS_i_sparse1 = torch.cat(lS_i_sparse1)
            output1 = lS_i_sparse1.new_empty(lS_i_sparse1.size())
            ind_req1 = ext_dist.dist.all_to_all_single(output1, lS_i_sparse1, async_op=True)

        if 'ly_sparse' not in cur_data[0]:
            with torch.autograd.profiler.record_function('Prof_sparse_emb_forward'):
                # sparse0 embeddings
                offset = torch.arange(batch_size0 * ext_dist.my_size)
                g_o_sparse0 = [offset for i in range(self.n_local_emb_sparse)]
                ly_sparse0 = self.apply_emb(dlrm.emb_sparse, g_o_sparse0, g_i_sparse0)
        else:
            ly_sparse0 = cur_data[0]['ly_sparse']

        if cur_data[1] and 'lS_i_sparse_res' not in cur_data[1]:
            ind_req1.wait()
            lS_i_sparse1 = output1.reshape(ext_dist.my_size, -1)
            g_i_sparse1 = [lS_i_sparse1[:, i * batch_size1:(i + 1) * batch_size1].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
        else:
            g_i_sparse1 = cur_data[1]['lS_i_sparse_res']


        with torch.autograd.profiler.record_function('Prof_alltoall_emb_forward'):
            a2a_req0 = ext_dist.alltoall(ly_sparse0, self.n_sparse_emb_per_rank)

        # dense0 embeddings
        with torch.autograd.profiler.record_function('Prof_dense_emb_forward'):
            ly_dense0 =  self.apply_emb(dlrm.emb_dense, lS_o0, lS_i_dense0)

        # bottom mlp 0
        with torch.autograd.profiler.record_function('Prof_bot_mlp_forward'):
            x0 = self.apply_mlp(dense_x0, self.bot_l)
        
        if cur_data[1]:
            # sparse1 embeddings
            with torch.autograd.profiler.record_function('Prof_sparse_emb_forward1'):
                offset = torch.arange(batch_size1 * ext_dist.my_size)
                g_o_sparse1 = [offset for i in range(self.n_local_emb_sparse)]
                ly_sparse1 = self.apply_emb(dlrm.emb_sparse, g_o_sparse1, g_i_sparse1)

        with torch.autograd.profiler.record_function('Prof_alltoall_emb_wait_forward'):
            ly_sparse0 = a2a_req0.wait()

        T_test_req0, T_test0 = ext_dist.all_gather_validation(T_test0, None)

        # concat emb data for split sparse0 embs
        ly_sparse0_full = []
        if ext_dist.my_size > len(self.ln_emb_sparse):
            for i in range(len(self.ln_emb_sparse)):
                ly_sparse0_split = torch.cat([ly_sparse0[j] for j in range(i, ext_dist.my_size, len(self.ln_emb_sparse))], 1)
                ly_sparse0_full.append(ly_sparse0_split)
        else:
            vector_lenght = 128
            for item in ly_sparse0:
                ly_sparse0_full += [item[:, emb_id * vector_lenght: (emb_id + 1) * vector_lenght] for emb_id in range(self.emb_sparse.n_tables)]
            ly_sparse0_full = [item.contiguous() for item in ly_sparse0_full]
            ly_sparse0_full = list(ly_sparse0_full)

        ly0 = list(ly_dense0) + ly_sparse0_full

        with torch.autograd.profiler.record_function('Prof_interaction_forward'):
            z0 = self.interact_features(x0, ly0)

        with torch.autograd.profiler.record_function('Prof_all_gather_t0'):
            T_test_req0.wait()

        if cur_data[1]:
            with torch.autograd.profiler.record_function('Prof_alltoall_emb_forward1'):
                a2a_req1 = ext_dist.alltoall(ly_sparse1, self.n_sparse_emb_per_rank)

        if next_data[0]:
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                next_lS_i_sparse0 = torch.cat([next_lS_i_sparse0 for _ in range(num_split_grps) ])
            next_lS_i_sparse0 = torch.cat(next_lS_i_sparse0)
            next_output0 = self.validation_output_ind0.get_data_base_on_input(next_lS_i_sparse0, next_lS_i_sparse0.size())
            next_ind_req0 = ext_dist.dist.all_to_all_single(next_output0, next_lS_i_sparse0, async_op=True)

        # top mlp 0
        with torch.autograd.profiler.record_function('Prof_top_mlp_forward'):
            p0 = self.apply_mlp(z0, self.top_l)

        if cur_data[1]:
            with torch.autograd.profiler.record_function('Prof_alltoall_emb_wait_forward1'):
                ly_sparse1 = a2a_req1.wait()

        if next_data[0]:
            next_ind_req0.wait()
            next_lS_i_sparse0 = next_output0.reshape(ext_dist.my_size, -1)
            next_g_i_sparse0 = [next_lS_i_sparse0[:, i * next_batch_size0:(i + 1) * next_batch_size0].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            next_data[0]['lS_i_sparse_res'] = next_g_i_sparse0

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0
        z0 = z0.float()

        Z_test_req0, Z_test0 = ext_dist.all_gather_validation(z0, None)

        if cur_data[1]:
            # dense1 embeddings
            with torch.autograd.profiler.record_function('Prof_dense_emb_forward1'):
                ly_dense1 =  self.apply_emb(dlrm.emb_dense, lS_o1, lS_i_dense1)
            # bottom mlp 1
            with torch.autograd.profiler.record_function('Prof_bot_mlp_forward1'):
                x1 = self.apply_mlp(dense_x1, self.bot_l)

        with torch.autograd.profiler.record_function('Prof_all_gather_z0'):
            Z_test_req0.wait()

        if cur_data[1]:
            T_test_req1, T_test1 = ext_dist.all_gather_validation(T_test1, None)

            # concat emb data for split sparse1 embs
            ly_sparse1_full = []
            vector_lenght = 128
            for item in ly_sparse1:
                ly_sparse1_full += [item[:, emb_id * vector_lenght: (emb_id + 1) * vector_lenght] for emb_id in range(self.emb_sparse.n_tables)]
            ly_sparse1_full = [item.contiguous() for item in ly_sparse1_full]
            ly_sparse1_full = list(ly_sparse1_full)

            ly1 = list(ly_dense1) + ly_sparse1_full

            with torch.autograd.profiler.record_function('Prof_interaction_forward1'):
                z1 = self.interact_features(x1, ly1)

        if next_data[1]:
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                next_lS_i_sparse1 = torch.cat([next_lS_i_sparse1 for _ in range(num_split_grps) ])
            next_lS_i_sparse1 = torch.cat(next_lS_i_sparse1)
            next_output1 = self.validation_output_ind1.get_data_base_on_input(next_lS_i_sparse1, next_lS_i_sparse1.size())
            next_ind_req1 = ext_dist.dist.all_to_all_single(next_output1, next_lS_i_sparse1, async_op=True)

        if cur_data[1]:
            p1 = self.apply_mlp(z1, self.top_l)

            with torch.autograd.profiler.record_function('Prof_all_gather_t1'):
                T_test_req1.wait()

            # clamp output if needed
            if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
                z1 = torch.clamp(
                    p1, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
                )
            else:
                z1 = p1
            z1 = z1.float()

        if next_data[1]:
            next_ind_req1.wait()
            next_lS_i_sparse1 = next_output1.reshape(ext_dist.my_size, -1)
            next_g_i_sparse1 = [next_lS_i_sparse1[:, i * next_batch_size1:(i + 1) * next_batch_size1].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            next_data[1]['lS_i_sparse_res'] = next_g_i_sparse1

        if cur_data[1]:
            Z_test_req1, Z_test1 = ext_dist.all_gather_validation(z1, None)

        if next_data[0]:
            offset = torch.arange(next_batch_size0 * ext_dist.my_size)
            next_g_o_sparse0 = [offset for i in range(self.n_local_emb_sparse)]
            next_ly_sparse0 = self.apply_emb(dlrm.emb_sparse, next_g_o_sparse0, next_data[0]['lS_i_sparse_res'])
            next_data[0]['ly_sparse'] = next_ly_sparse0

        if cur_data[1]:
            Z_test_req1.wait()

        if cur_data[1]:
            return (Z_test0, T_test0), (Z_test1, T_test1)
        else:
            return (Z_test0, T_test0), None


    def distributed_forward(self, dense_x, lS_o, lS_i, is_train=False):
        local_batch_size = dense_x.size()[0]

        #vector_lenght = self.emb_l.weights[0].size()[1]
        # WARNING: # of ranks must be <= batch size in distributed_forward call
        if local_batch_size < ext_dist.my_size:
            sys.exit("ERROR: local_batch_size (%d) must be larger than number of ranks (%d)" % (local_batch_size, ext_dist.my_size))
        
        rank_id = dlrm.rank
        #print("#####################rank_id:", rank_id)
        if not isinstance(lS_i, list):
            lS_i_dense = [lS_i[i] for i in self.ln_emb_dense]
            lS_i_sparse = [lS_i[i] for i in self.ln_emb_sparse]

        lS_o_dense = [lS_o[i] for i in self.ln_emb_dense]
        lS_o_sparse = [lS_o[i] for i in self.ln_emb_sparse]

        global first_iteration_for_train
        if first_iteration_for_train or not is_train: 
            #lS_i_sparse = ext_dist.shuffle_data(lS_i_sparse)
            lS_i_sparse = torch.cat(lS_i_sparse)
            output_lS_i_sparse = lS_i_sparse.new_empty(lS_i_sparse.size())
            #print("cat time:", 1000*(time.time() - shuffle_start))
            req = ext_dist.dist.all_to_all_single(output_lS_i_sparse, lS_i_sparse)
            lS_i_sparse = output_lS_i_sparse.reshape(ext_dist.my_size, -1)
            g_i_sparse = [lS_i_sparse[:, i * local_batch_size:(i + 1) * local_batch_size].reshape(-1) for i in range(len(self.local_ln_emb_sparse))]
            if is_train:
                first_iteration_for_train = False
        else:
            g_i_sparse = lS_i[1]
            lS_i_dense = lS_i[0]

        offset = torch.arange(local_batch_size * ext_dist.my_size)
        #print("shuffle time:", 1000*(shuffle_end - shuffle_start))

        g_o_sparse = [offset for i in range(self.n_local_emb_sparse)]
        ly_sparse = self.apply_emb(dlrm.emb_sparse, g_o_sparse, g_i_sparse)
        a2a_req = ext_dist.alltoall(ly_sparse, self.n_sparse_emb_per_rank)

        #dense embedding 
        ly_dense =  self.apply_emb(dlrm.emb_dense, lS_o_dense, lS_i_dense)
        # bottom mlp
        x = self.apply_mlp(dense_x, self.bot_l)
        # ext_dist.barrier()
        ly_sparse = a2a_req.wait()
        _ly = []
        vector_lenght = 128
        for item in ly_sparse:
            _ly += [item[:, emb_id * vector_lenght: (emb_id + 1) * vector_lenght] for emb_id in range(self.emb_sparse.n_tables)]
        _ly = [item.contiguous() for item in _ly]
        _ly = list(ly_dense) + list(_ly)

        a2a_ind_req = None #ovlerlap the a2a_ind_req with interaction/top_mlp

        # interactions
        z = self.interact_features(x, _ly)
        # top mlp
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(
                p, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )                                                                         
        else:
            z = p
        if is_train:
            with record_function("dist_data_loader"):
                start = time_wrap()
                # ext_dist.barrier()
                load_data(buffer_num, args.bf16)
                #get global index for sparse embedding
                (X_next, lS_o_next, lS_i_next, T_next, W_next, CBPP_next) = data_buffer[0]
                lS_i_dense_next = [lS_i_next[i] for i in self.ln_emb_dense]
                lS_i_sparse_next = [lS_i_next[i] for i in self.ln_emb_sparse]
                if ext_dist.my_size > len(self.ln_emb_sparse):
                    num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                    lS_i_sparse_next = torch.cat([lS_i_sparse_next for _ in range(num_split_grps) ])
                lS_i_sparse_next = torch.cat(lS_i_sparse_next)
                output_lS_i_sparse_next = lS_i_sparse_next.new_empty(lS_i_sparse_next.size())
                a2a_ind_req = ext_dist.dist.all_to_all_single(output_lS_i_sparse_next, lS_i_sparse_next, async_op=True)
                a2a_ind_req.wait()
                lS_i_sparse_next = output_lS_i_sparse_next.reshape(ext_dist.my_size, -1)
                batch_size_next = X_next.size()[0]
                g_i_sparse_next = [lS_i_sparse_next[:, i * batch_size_next:(i + 1) * batch_size_next].reshape(-1) for i in range(len(self.local_ln_emb_sparse))]
                data_buffer[0] = (X_next, lS_o_next, [lS_i_dense_next, g_i_sparse_next], T_next, W_next, CBPP_next)
                # ext_dist.barrier()
                self.load_data_time += time_wrap() - start
        return z
 

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(self.emb_l, lS_o, lS_i) 
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p
       
        return z


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def trace_model(args, dlrm, test_ld):
    dlrm.eval()
    for j, inputBatch in enumerate(test_ld):
        X, lS_o, lS_i, _, _, _ = unpack_batch(inputBatch)
        if args.bf32:
            ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
        if args.bf16:
            # at::GradMode::is_enabled() will query a threadlocal flag
            # but new thread generate from throughputbench mark will 
            # init this flag to true, so we temporal cast embedding's
            # weight to bfloat16 for now
            if args.inference_only and args.ipex_interaction:
                dlrm.emb_l.bfloat16()
            if args.ipex_interaction:
                dlrm = ipex.optimize(dlrm, dtype=torch.bfloat16, inplace=True)
            else:
                torch._C._jit_set_autocast_mode(False)
        elif args.int8 and args.ipex_interaction:
            from intel_extension_for_pytorch.quantization import prepare, convert
            if args.num_cpu_cores != 0:
                torch.set_num_threads(args.num_cpu_cores)
            qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
            prepare(dlrm, qconfig, example_inputs=(X, lS_o, lS_i), inplace=True)
            dlrm.load_qconf_summary(qconf_summary = args.int8_configure)
            convert(dlrm, inplace=True)
        elif args.ipex_interaction:
            if args.bf32:
                ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
            dlrm = ipex.optimize(dlrm, dtype=torch.float, inplace=True, auto_kernel_selection=True)
        if args.int8:
            print("Start to trace/freeze for int8, may need {} to save int8 weight".format(dlrm.numel / 1024 / 1024 / 1024))
            print("Current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            if args.ipex_interaction:
                dlrm = torch.jit.trace(dlrm, [X, lS_o, lS_i])
                dlrm = torch.jit.freeze(dlrm)
            print("After trace/freeze, current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            dlrm(X, lS_o, lS_i)
            dlrm(X, lS_o, lS_i)
        else:
            if args.ipex_interaction:
                with torch.cpu.amp.autocast(enabled=args.bf16):
                    dlrm = torch.jit.trace(dlrm, (X, lS_o, lS_i), check_trace=True)
                    dlrm = torch.jit.freeze(dlrm)
        # torch.compile() path
        if args.inductor:
            from torch._inductor import config as inductor_config
            inductor_config.cpp_wrapper = True
            if args.int8:
                from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
                import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
                from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
                from torch.export import export_for_training
                print('[Info] Running torch.compile() INT8 quantization')
                with torch.no_grad():
                    example_inputs = (X, lS_o, lS_i)
                    exported_model = export_for_training(
                        dlrm,
                        example_inputs
                    ).module()
                    quantizer = X86InductorQuantizer()
                    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                    prepared_model = prepare_pt2e(exported_model, quantizer)
                    prepared_model(example_inputs)
                    converted_model = convert_pt2e(prepared_model)
                    torch.ao.quantization.move_exported_model_to_eval(converted_model)
                    print('[Info] Running torch.compile() with default backend')
                    dlrm = torch.compile(dlrm)
            elif args.bf16:
                with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
                    print('[Info] Running torch.compile() with default backend')
                    dlrm = torch.compile(dlrm)
            else:
                with torch.no_grad():
                    print('[Info] Running torch.compile() with default backend')
                    dlrm = torch.compile(dlrm)
        with torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16):
            dlrm(X, lS_o, lS_i)
            dlrm(X, lS_o, lS_i)
        return dlrm


def run_throughput_benchmark(args, dlrm, test_ld):
    if args.num_cpu_cores != 0:
        torch.set_num_threads(1)
    bench = ThroughputBenchmark(dlrm)
    for j, inputBatch in enumerate(test_ld):
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        bench.add_input(X, lS_o, lS_i)
        if j == 1000: 
            break
    stats = bench.benchmark(
        num_calling_threads=args.share_weight_instance,
        num_warmup_iters=100,
        num_iters=args.num_batches * args.share_weight_instance,
    )
    print(stats)
    latency = stats.latency_avg_ms
    throughput = (1 / latency) * 1000 * args.test_mini_batch_size * args.share_weight_instance
    print("Throughput: {:.3f} fps".format(throughput))
    sys.exit()


def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
):
    test_accu = 0
    test_samp = 0

    if args.print_auc:
        scores = []
        targets = []

    total_time = 0
    total_iter = 0
    if args.inference_only:
        dlrm = trace_model(args, dlrm, test_ld)
    if args.share_weight_instance != 0:
        run_throughput_benchmark(args, dlrm, test_ld)
    with torch.cpu.amp.autocast(enabled=args.bf16):
        if ext_dist.my_size > 1:
            i = 0
            n_test_iter = 2     # exec 2 iter per validation
            cur_data = [dict() for _ in range(n_test_iter)]
            next_data = [dict() for _ in range(n_test_iter)]
            n = nbatches_test // n_test_iter
            batch_remainder = nbatches_test % n_test_iter
            
            while i < n:
                # load 2 iter
                if i == 0:
                    cur_batch_0 = test_ld.dataset[i]
                    cur_batch_1 = test_ld.dataset[i + 1]

                    X_test0, lS_o_test0, lS_i_test0, T_test0, _, _ = unpack_batch(
                        cur_batch_0
                    )
                    X_test1, lS_o_test1, lS_i_test1, T_test1, _, _ = unpack_batch(
                        cur_batch_1
                    )
                    cur_data[0]['dense_x'] = X_test0
                    cur_data[0]['T_test'] = T_test0
                    cur_data[0]['lS_i'] = lS_i_test0
                    cur_data[0]['lS_o'] = lS_o_test0

                    cur_data[1]['dense_x'] = X_test1
                    cur_data[1]['T_test'] = T_test1
                    cur_data[1]['lS_i'] = lS_i_test1
                    cur_data[1]['lS_o'] = lS_o_test1
                else:
                    for j in range(n_test_iter):
                        cur_data[j] = next_data[j]
                # preload 2 iter
                if i + 1 < n:
                    next_batch_0 = test_ld.dataset[(i+1)*n_test_iter]
                    next_batch_1 = test_ld.dataset[(i+1)*n_test_iter + 1]

                    X_test0, lS_o_test0, lS_i_test0, T_test0, _, _ = unpack_batch(
                        next_batch_0
                    )
                    X_test1, lS_o_test1, lS_i_test1, T_test1, _, _ = unpack_batch(
                        next_batch_1
                    )
                    next_data[0] = {'dense_x': X_test0, 'T_test': T_test0, 'lS_i': lS_i_test0, 'lS_o': lS_o_test0}

                    next_data[1] = {'dense_x': X_test1, 'T_test': T_test1, 'lS_i': lS_i_test1, 'lS_o': lS_o_test1}
                elif batch_remainder == 1:
                    next_batch_0 = test_ld.dataset[len(test_ld) - 1]
                    X_test0, lS_o_test0, lS_i_test0, T_test0, _, _ = unpack_batch(
                        next_batch_0
                    )
                    next_data[0] = {'dense_x': X_test0, 'T_test': T_test0, 'lS_i': lS_i_test0, 'lS_o': lS_o_test0}
                    next_data[1] = None

                res_pack = dlrm.distributed_forward_2iter_overlap(cur_data, next_data)
                if i == n - 1:
                    last_Z_test = dlrm(next_data[0]['dense_x'], next_data[0]['lS_o'], next_data[0]['lS_i'])
                if args.print_auc:
                    for j in range(n_test_iter):
                        if res_pack[j]:
                            Z_test, T_test = res_pack[j]
                            S_test = Z_test.detach().cpu().float()  # numpy array
                            T_test = T_test.detach().cpu().float()  # numpy array
                            scores.append(S_test)
                            targets.append(T_test)
                    if i == n - 1:
                        S_test = last_Z_test.detach().cpu().float()  # numpy array
                        T_test = T_test0.detach().cpu().float()  # numpy array
                        scores.append(S_test)
                        targets.append(T_test)
                i += 1
            
        else:
            for i, testBatch in enumerate(test_ld):
                should_print = ((i + 1) % args.print_freq == 0 or i + 1 == len(test_ld)) and args.inference_only
                if should_print:
                    gT = 1000.0 * total_time / total_iter
                    print(
                        "Finished {} it {}/{}, {:.2f} ms/it,".format(
                            "inference", i + 1, len(test_ld), gT
                        ),
                        flush=True,
                    )
                    total_time = 0
                    total_iter = 0
                # early exit if nbatches was set by the user and was exceeded
                if args.inference_only and nbatches > 0 and i >= nbatches:
                    break

                X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
                    testBatch
                )

                # forward pass
                start = time_wrap()
                Z_test = dlrm(X_test, lS_o_test, lS_i_test)

                total_time += (time_wrap() - start)
                total_iter += 1

                if args.print_auc:
                    if ext_dist.my_size > 1:
                        Z_test = ext_dist.all_gather(Z_test, None)
                        T_test = ext_dist.all_gather(T_test, None)
                    S_test = Z_test.detach().cpu().float().numpy()  # numpy array
                    T_test = T_test.detach().cpu().float().numpy()  # numpy array
                    scores.append(S_test)
                    targets.append(T_test)
                elif not args.inference_only:
                    with record_function("DLRM accuracy compute"):
                        # compute loss and accuracy
                        S_test = Z_test.detach().cpu().float().numpy()  # numpy array
                        T_test = T_test.detach().cpu().float().numpy()  # numpy array

                        mbs_test = T_test.shape[0]  # = mini_batch_size except last
                        A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

                        test_accu += A_test
                        test_samp += mbs_test
                else:
                    # do nothing to save time
                    pass

    if args.print_auc:
        with record_function("DLRM mlperf sklearn metrics compute"):
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)

            if args.print_recall_precision_f1_ap:
                metrics = {
                    "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
                        y_true=y_true, y_pred=np.round(y_score)
                    ),
                    "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
                        y_true=y_true, y_pred=np.round(y_score)
                    ),
                    "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
                        y_true=y_true, y_pred=np.round(y_score)
                    ),
                    "ap": sklearn.metrics.average_precision_score,
                }
        roc_auc, _, accuracy = ipex._C.roc_auc_score_all(torch.Tensor(targets), torch.Tensor(scores))
        validation_results = {}
        validation_results["roc_auc"] = roc_auc
        validation_results["accuracy"] = accuracy
        acc_test = validation_results["accuracy"]
        if args.print_recall_precision_f1_ap:
            for metric_name, metric_function in metrics.items():
                validation_results[metric_name] = metric_function(targets, scores)
    elif not args.inference_only:
        acc_test = test_accu / test_samp
    else:
        pass

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
    }
    if not args.inference_only:
        model_metrics_dict["test_acc"] = acc_test

    if args.print_auc:
        is_best = validation_results["roc_auc"] > best_auc_test
        if is_best:
            best_auc_test = validation_results["roc_auc"]
            model_metrics_dict["test_auc"] = best_auc_test

        result_fmt_str = (" auc {:.4f}, best auc {:.4f},".format(
                validation_results["roc_auc"], best_auc_test
            )
            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                validation_results["accuracy"] * 100, best_acc_test * 100
            ))
        if args.print_recall_precision_f1_ap:
            result_fmt_str =  ("recall {:.4f}, precision {:.4f},".format(
                validation_results["recall"],
                validation_results["precision"],
            )
            + " f1 {:.4f}, ap {:.4f},".format(
                validation_results["f1"], validation_results["ap"]
            )) + result_fmt_str
        if (ext_dist.my_size > 1 and ext_dist.dist.get_rank() == 1) or ext_dist.my_size <= 1:
            print(
                result_fmt_str,
                flush=True,
            )
            print("Accuracy: {:.34} ".format(validation_results["roc_auc"]))
        if not args.inference_only:
            if args.mlperf_auc_threshold != 0.0 and best_auc_test > args.mlperf_auc_threshold:
                train_end = time.time()
                print("Have reached the auc threshold:", args.mlperf_auc_threshold, ", stop training")
    elif not args.inference_only:
        is_best = acc_test > best_acc_test
        if is_best:
            best_acc_test = acc_test
        print(
            " accuracy {:3.3f} %, best {:3.3f} %".format(
                acc_test * 100, best_acc_test * 100
            ),
            flush=True,
        )
    else:
        pass
    if not args.inference_only:
        return model_metrics_dict, is_best
    else:
        return


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # training
    parser.add_argument("--mini-batch-size", type=int, default=-1)
    parser.add_argument("--local-batch-size", type=int, default=-1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=6400)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    # intel
    parser.add_argument("--print-auc", action="store_true", default=False)
    parser.add_argument("--print-recall-precision-f1-ap", action="store_true", default=False)
    parser.add_argument("--should-test", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--bf32", action="store_true", default=False)
    parser.add_argument("--share-weight-instance", type=int, default=0)
    parser.add_argument("--num-cpu-cores", type=int, default=0)
    parser.add_argument("--ipex-interaction", action="store_true", default=False)
    parser.add_argument("--ipex-merged-emb", action="store_true", default=False)
    parser.add_argument("--num-warmup-iters", type=int, default=1000)
    parser.add_argument("--int8", action="store_true", default=False)
    parser.add_argument("--calibration", action="store_true", default=False)
    parser.add_argument("--int8-configure", type=str, default="./int8_configure.json")
    parser.add_argument("--dist-backend", type=str, default="ccl")
     # embedding table is sparse table only if sparse_dense_boundary >= 2048
    parser.add_argument("--sparse-dense-boundary", type=int, default=2048)
    parser.add_argument("--hybrid-gradient-emb", action="store_true", default=False)
    parser.add_argument('--inductor', action='store_true', default=False,
                        help='using torch.compile()')

    global args
    global nbatches
    global nbatches_test
    args = parser.parse_args()
    
    print(args)
    if args.ipex_interaction:
        print('Using ipex')
        import intel_extension_for_pytorch as ipex
        global ipex
    ext_dist.init_distributed(backend=args.dist_backend)
    if args.mini_batch_size < 0:
        if args.local_batch_size > 0:
            assert ext_dist.my_size > 1
            args.mini_batch_size = args.local_batch_size * ext_dist.my_size
        else:
            assert args.test_mini_batch_size > 0
            args.mini_batch_size = args.test_mini_batch_size

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size

    device = torch.device("cpu")
    print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    global train_ld
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
    nbatches_test = len(test_ld)

    ln_emb = train_data.counts
    # enforce maximum limit on number of vectors per embedding
    if args.max_ind_range > 0:
        ln_emb = np.array(
            list(
                map(
                    lambda x: x if x < args.max_ind_range else args.max_ind_range,
                    ln_emb,
                )
            )
        )
    else:
        ln_emb = np.array(ln_emb)
    m_den = train_data.m_den
    ln_bot[0] = m_den

    args.ln_emb = ln_emb.tolist()

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    # approach 1: all
    # num_int = num_fea * num_fea + m_den_out
    # approach 2: unique
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out

    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global dlrm
    np_init_emb_weight = not args.inference_only
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_threshold=args.loss_threshold,
        np_init_emb_weight=np_init_emb_weight,
    )
    if args.ipex_merged_emb:
        if ext_dist.my_size > 1:
            dlrm.emb_sparse = ipex.nn.modules.MergedEmbeddingBagWithSGD.from_embeddingbag_list(dlrm.emb_sparse, lr=args.learning_rate/ext_dist.my_size)
            dlrm.emb_dense = ipex.nn.modules.MergedEmbeddingBag.from_embeddingbag_list(dlrm.emb_dense)
        else:
            dlrm.emb_l = ipex.nn.modules.MergedEmbeddingBagWithSGD.from_embeddingbag_list(dlrm.emb_l, lr=args.learning_rate)

    if not args.inference_only:
        #ToDo: may cause convergence issue for sparse table lr
        optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            args.lr_num_warmup_steps,
            args.lr_decay_start_step,
            args.lr_num_decay_steps,
        )

    ### main loop ###
    # training or inference
    best_acc_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        print("Loading weight will minimally cost ~{} G memory".format(dlrm.numel * 4 / 1024 / 1024 / 1024))
        print("Current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
        print("Loading weight done, current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_acc_test = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0
        print("Copy loaded state to model done, current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        del(ld_model)
        print("Remove loaded state, current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}".format(
                ld_train_loss,
            )
        )
        print("Testing state: accuracy = {:3.3f} %".format(ld_acc_test * 100))

    ext_dist.barrier()

    if args.calibration:
        assert args.load_model != "", "need load weight to do calibration"
        dlrm.eval()
        qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
            weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
        for j, inputBatch in enumerate(train_ld):
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
            example_inputs = (X, lS_o, lS_i)
            prepared_dlrm = prepare(dlrm, qconfig, example_inputs=example_inputs, inplace=True)
            break

        for j, inputBatch in enumerate(train_ld):
            prepared_dlrm(X, lS_o, lS_i)
            if j == 64:
                break
        prepared_dlrm.save_qconf_summary(qconf_summary = args.int8_configure)
        print("calibration done, save config file to ", args.int8_configure)
        exit()

    print("time/loss/accuracy (if enabled):")
    
    global buffer_num
    if ext_dist.my_size > 1:
        buffer_num = 1
    global data_buffer
    data_buffer = buffer_num * [None]
    global data_iter
    data_iter = iter(train_ld)
    buffer_num = buffer_num if buffer_num <= nbatches else nbatches
    # data_load_begin = time.time()
    load_data(buffer_num, args.bf16)
    # print(buffer_num, ": data item loaded, data_load_time is {:.6f}s".format(time.time() - data_load_begin))

        
    if not args.inference_only:
        X, lS_o, lS_i, T = train_ld.dataset.__getitem__(0)
        lS_i = lS_i.contiguous()
        sample_input = (X, lS_o, lS_i)
        if args.bf16:
            print("Start to split weight to bf16 and trail part, or saving whole fp32 master weight, create bf16 weight copy")
            print("Maximum will use ~ {} G memory, may use less memory if useless fp32 weight (for split path) will be released in time ".format(dlrm.numel * 4 / 1024 / 1024 /1024))
            if args.ipex_merged_emb:
                dlrm, optimizer = ipex.optimize(dlrm, dtype=torch.bfloat16, optimizer=optimizer, inplace=True, sample_input=sample_input)
                print("args.ipex_merged_emb:", args.ipex_merged_emb)
                print("###############Current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
                if ext_dist.my_size > 1:
                    dlrm.emb_sparse.to_bfloat16_train()
                else:
                    dlrm.emb_l.to_bfloat16_train()
            print("Weight cast done, current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        else:
            if args.ipex_merged_emb:
                dlrm, optimizer = ipex.optimize(dlrm, dtype=torch.float, optimizer=optimizer, inplace=True, sample_input=sample_input, auto_kernel_selection=True)
                if args.bf32:
                    ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")

        for i in range(len(dlrm.top_l)):
            if args.ipex_merged_emb:
                if isinstance(dlrm.top_l[i], ipex.nn.utils._weight_prepack._IPEXLinear):
                    if isinstance(dlrm.top_l[i+1], torch.nn.ReLU):
                        dlrm.top_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.top_l[i], 'relu')
                    else:
                        dlrm.top_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.top_l[i], 'sigmoid')
                    dlrm.top_l[i + 1] = torch.nn.Identity()
        for i in range(len(dlrm.bot_l)):
            if args.ipex_merged_emb:
                if isinstance(dlrm.bot_l[i], ipex.nn.utils._weight_prepack._IPEXLinear):
                    if isinstance(dlrm.bot_l[i+1], torch.nn.ReLU):
                        dlrm.bot_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.bot_l[i], 'relu')
                    else:
                        dlrm.bot_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.bot_l[i], 'sigmoid')
                    dlrm.bot_l[i + 1] = torch.nn.Identity()

        if ext_dist.my_size > 1:
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l, gradient_as_bucket_view=True, broadcast_buffers=False, find_unused_parameters=True)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l, gradient_as_bucket_view=True, broadcast_buffers=False, find_unused_parameters=True)
            dlrm.emb_dense = ext_dist.DDP(dlrm.emb_dense, gradient_as_bucket_view=True, broadcast_buffers=False)

        if args.inductor:
            with torch.cpu.amp.autocast(enabled=args.bf16):
                print('[Info] Running training steps torch.compile() with default backend')
                dlrm = torch.compile(dlrm)
    training_record = [0, 0]
    def update_training_performance(time, iters, training_record=training_record):
        if iters > args.num_warmup_iters:
            training_record[0] += time
            training_record[1] += 1

    def print_training_performance( training_record=training_record):
        if training_record[0] == 0:
            print("num-batches larger than warm up iters, please increase num-batches or decrease warmup iters")
            exit()
        if args.local_batch_size > 0:
            total_samples = training_record[1] * args.local_batch_size
        else:
            total_samples = training_record[1] * args.mini_batch_size
        throughput = total_samples / training_record[0] * 1000
        print("Throughput: {:.3f} fps".format(throughput))

    test_freq = args.test_freq if args.test_freq != -1  else nbatches // 20
    print("Initialize for not inference only done, current mem usage: {} G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    # print(dlrm)
    train_start = time.time()
    total_train_time_wo_dl_eval = 0
    wait_it = 0
    warmup_it = 400
    active_it = 20
    lrs =[]


    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        if ext_dist.my_size > 1:
            rank = ext_dist.dist.get_rank()
            prof.export_chrome_trace("dlrm_training_trace_rank_{}_step{}.json".format(rank,str(prof.step_num)))
        else:
            rank = 0
            prof.export_chrome_trace("dlrm_training_trace_step_{}.json".format(str(prof.step_num)))

        file_prefix = "%s/dlrm_s_pytorch_r%d" % (".", rank)
        #with open("dlrm_s_pytorch.prof", "w") as prof_f:
        with open("%s.prof" % file_prefix, "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            wait=wait_it,
            warmup=warmup_it,
            active=active_it),
        on_trace_ready=trace_handler
        ) as prof:
        if not args.inference_only:
            k = 0
            while k < args.nepochs:

                if k < skip_upto_epoch:
                    continue
                
                j = 0
                # for j, inputBatch in enumerate(train_ld):
                while j < nbatches: 
                    # if j < skip_upto_batch:
                    #     continue

                    for d in range(buffer_num):
                        # X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
                        (X, lS_o, lS_i, T, W, CBPP) = data_buffer[d]
                        
                        t1 = time_wrap()

                        # early exit if nbatches was set by the user and has been exceeded
                        if nbatches > 0 and j >= nbatches:
                            break

                        mbs = T.shape[0]  # = args.mini_batch_size except maybe for last

                        # forward pass

                        if args.ipex_merged_emb and hasattr(dlrm, 'emb_l') and isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                            dlrm.emb_l.sgd_args = dlrm.emb_l.sgd_args._replace(lr=lr_scheduler.get_last_lr()[0])

                        if args.ipex_merged_emb and hasattr(dlrm, 'emb_sparse') and isinstance(dlrm.emb_sparse, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                            dlrm.emb_sparse.sgd_args = dlrm.emb_sparse.sgd_args._replace(lr=lr_scheduler.get_last_lr()[0]/ext_dist.my_size)

                        with torch.cpu.amp.autocast(enabled=args.bf16):
                            Z = dlrm_wrap(
                                X,
                                lS_o,
                                lS_i,
                                True,
                            ).float()
                        # losis
                        E = loss_fn_wrap(Z, T)

                        # compute loss and accuracy
                        L = E.detach().cpu().numpy()  # numpy array

                        with record_function("DLRM backward"):
                            # scaled error gradient propagation
                            # (where we do not accumulate gradients across mini-batches)
                            optimizer.zero_grad(set_to_none=True)
                            # backward pass
                            E.backward()
                        with record_function("DLRM update"):
                            # optimizer
                            optimizer.step()

                        lr_scheduler.step()

                        t2 = time_wrap() - dlrm.load_data_time
                        dlrm.load_data_time = 0
                        total_train_time_wo_dl_eval += (t2 - t1)
                        total_time += t2 - t1

                        total_loss += L * mbs
                        total_iter += 1
                        total_samp += mbs
                        if args.enable_profiling:
                            prof.step()
                            if j >=  wait_it + warmup_it + active_it:
                                break
                        should_print = ((j + 1) % args.print_freq == 0) or (
                            j + 1 == nbatches
                        )
                        should_test = (
                            (args.should_test)
                            and (((j + 1) % test_freq == 0) or (j + 1 == nbatches))
                        )

                        # print time, loss and accuracy
                        if should_print or should_test:
                            gT = 1000.0 * total_time / total_iter if args.print_time else -1
                            total_time = 0

                            train_loss = total_loss / total_samp
                            total_loss = 0

                            str_run_type = (
                                "inference" if args.inference_only else "training"
                            )

                            wall_time = ""
                            if args.print_wall_time:
                                wall_time = " ({})".format(time.strftime("%H:%M"))
                            # if ext_dist.my_size > 1 and ext_dist.dist.get_rank() ==1 :
                            print(
                                "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                                    str_run_type, j + 1, nbatches, k, gT
                                )
                                + " loss {:.6f}".format(train_loss)
                                + wall_time,
                                flush=True,
                            )
                            update_training_performance(gT, j)

                            total_iter = 0
                            total_samp = 0

                        # testing
                        if should_test:
                            eval_begin = time.time()
                            model_metrics_dict, is_best = inference(
                                args,
                                dlrm,
                                best_acc_test,
                                best_auc_test,
                                test_ld,
                            )
                            if is_best:
                                best_auc_test = model_metrics_dict["test_auc"]
                                best_acc_test = model_metrics_dict["test_acc"]
                            eval_end = time.time()
                            dlrm.load_data_time = 0
                            if ext_dist.my_size > 1 and ext_dist.dist.get_rank() ==1 :
                                print("Evauation at {} iteration using {} s".format(j+1, eval_end- eval_begin))
                            if (
                                is_best
                                and not (args.save_model == "")
                                and not args.inference_only
                            ):
                                model_metrics_dict["epoch"] = k
                                model_metrics_dict["iter"] = j + 1
                                model_metrics_dict["train_loss"] = train_loss
                                model_metrics_dict["total_loss"] = total_loss
                                model_metrics_dict[
                                    "opt_state_dict"
                                ] = optimizer.state_dict()
                                print("Saving model to {}".format(args.save_model))
                                torch.save(model_metrics_dict, args.save_model)

                            if (
                                (args.mlperf_auc_threshold > 0)
                                and (best_auc_test > args.mlperf_auc_threshold)
                            ):
                                train_end = time.time()
                                if ext_dist.dist.get_rank() == 1:
                                    print("The TTT w/ dataloader and evaluation is {} mins".format((train_end - train_start)/60.0))
                                    print("The TTT w/o dataloader and evaluation is {} mins".format((total_train_time_wo_dl_eval)/60.0))
                                exit()
                        j += 1

                    if ext_dist.my_size == 1:
                        buffer_num = buffer_num if (nbatches - j) > buffer_num else (nbatches - j)
                        next_epoch = buffer_num == 0 or load_data(buffer_num, args.bf16)
                        if next_epoch:
                            k += 1
                            if k >= args.nepochs:
                                break

                k += 1  # nepochs
        else:
            print("Testing for inference only")
            with torch.no_grad():
                inference(
                    args,
                    dlrm,
                    best_acc_test,
                    best_auc_test,
                    test_ld
                )

    # profiling
    print_training_performance()
    # workaround oneccl bad termination issue.
    if ext_dist.my_size > 1:
        ext_dist.dist.destroy_process_group()

    if args.enable_profiling:
        time_stamp = str(datetime.datetime.now()).replace(" ", "_")
        with open("dlrm_s_pytorch" + time_stamp + "_shape.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cpu_time_total"
                )
            )
        with open("dlrm_s_pytorch" + time_stamp + "_total.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("dlrm_s_pytorch" + time_stamp + ".json")

if __name__ == "__main__":
    run()
