#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import os
from argparse import ArgumentParser
from dataloader import MAG, MAG240M, Products, Arxiv, Papers100M
from sampler import Sampler

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn_mag",
                        help="ogbn_mag, ogbn_mag240m, ogbn_products, ogbn_papers100M, or ogbn_arxiv")
    parser.add_argument("--ogb_dir", type=str, default="/tmp/rgat_dataset/",
                        help="directory that ogb datasets are saved in")
    parser.add_argument("--graph_schema_dir", type=str, required=True,
                        help="directory containing graph schemas (.json, .pbtxt) for sampling the data")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir, defaults to None")
    args = parser.parse_args()

    graph_schema_path = os.path.join(args.graph_schema_dir, f"{args.dataset}_subgraph_schema.pbtxt")
    sampling_schema_path = os.path.join(args.graph_schema_dir, f"{args.dataset}_sampling_schema.json")
    output_dir = args.graph_schema_dir
    if args.output_dir is None:
        data_dir = os.path.join(args.ogb_dir, args.dataset)
    else:
        data_dir = args.output_dir

    if args.dataset == "ogbn_mag":
        dataloader = MAG(ogb_dir=args.ogb_dir,
                         output_dir=output_dir,
                         schema_path=graph_schema_path)

    # Products is the same as the base Dataloader class and can be used for arxiv as well
    elif args.dataset == "ogbn_products":
        dataloader = Products(ogb_dir=args.ogb_dir,
                              output_dir=output_dir,
                              schema_path=graph_schema_path)

    elif args.dataset == "ogbn_arxiv":
        dataloader = Arxiv(ogb_dir=args.ogb_dir,
                           output_dir=output_dir,
                           schema_path=graph_schema_path)

    elif args.dataset == "ogbn_mag240m":
        dataloader = MAG240M(ogb_dir=args.ogb_dir,
                             output_dir=output_dir,
                             schema_path=graph_schema_path)

    else:
        dataloader = Papers100M(ogb_dir=args.ogb_dir,
                                output_dir=output_dir,
                                schema_path=graph_schema_path)

    print("creating sampler")
    sampler = Sampler(dataloader=dataloader,
                      sampling_schema_path=sampling_schema_path)
    print("sampler created")
    # for split in ["train", "valid"]:
    for split in ["test"]:
        print("split: ", split)
        sampler.sample_dataset(split=split, dir=data_dir)
    print("The dataset is saved at: ", data_dir)
