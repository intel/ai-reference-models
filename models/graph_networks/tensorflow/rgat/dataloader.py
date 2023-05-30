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
import time
import numpy as np
import tensorflow_gnn as tfgnn
from scipy.sparse import csr_matrix, csc_matrix
from ogb.nodeproppred import NodePropPredDataset
from ogb.lsc import MAG240MDataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
The Dataloader class loads an OGB dataset and converts it to a
NetworkX graph to be used for sampling.

Init Args:
    dataset: name of an OGB dataset (current only tested on MAG and Products)
    ogb_dir: directory where the OGB dataset is or will be saved
    output_dir: directory where the Networkx graphs will be saved
    schema_path: path to the TFGNN GraphTensor Schema
'''


class Dataloader():
    def __init__(self,
                 dataset: str,
                 ogb_dir: str,
                 output_dir: str,
                 schema_path: str) -> None:
        self.output_dir = output_dir
        graph_schema = tfgnn.read_schema(schema_path)
        self.graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
        print("Loading OGB dataset...", end="\r")
        start = time.time()
        dataset = NodePropPredDataset(name=dataset, root=ogb_dir)
        elapsed = time.time() - start
        print(f"Loaded OGB dataset ({elapsed:0.2f} seconds)")

        self.name = dataset.name
        self.root = dataset.root
        self.meta_info = dataset.meta_info

        self.splits = dataset.get_idx_split()
        self.num_classes = dataset.num_classes

        graph, labels = dataset[0]
        # convert a homogeneous graph to the same format as hetero graphs
        if self.meta_info["is hetero"] == "False":
            edge_set = self._get_edge_sets_from_spec()[0]
            node_set = self._get_node_sets_from_spec()[0]
            self.num_nodes_dict = {node_set: graph["num_nodes"]}
            self.node_feat_dict = {node_set: graph["node_feat"]}
            self.edge_index_dict = {edge_set: graph["edge_index"]}
            self.labels = {node_set: labels}
            for split in self.splits.keys():
                self.splits[split] = {node_set: self.splits[split]}
        else:
            self.num_nodes_dict = graph['num_nodes_dict']
            self.node_feat_dict = graph["node_feat_dict"]
            self.edge_index_dict = graph["edge_index_dict"]
            self.labels = labels

        print("Converting to networkx graphs...", end="\r")
        start = time.time()
        # self.nx_graphs = self._ogb_to_nx()
        self.sparse_graphs_csr, self.sparse_graphs_csc = self._ogb_to_sparse()
        elapsed = time.time() - start
        print(f"Converted to networkx graphs ({elapsed:0.2f} seconds)")

    def _get_edge_sets_from_spec(self,):
        edge_sets = []
        for es in list(self.graph_spec.edge_sets_spec.keys()):
            source_type = self.graph_spec.edge_sets_spec[es].adjacency_spec.source_name
            target_type = self.graph_spec.edge_sets_spec[es].adjacency_spec.target_name
            edge_sets.append((source_type, es, target_type))
        return edge_sets

    def _get_node_sets_from_spec(self,):
        return list(self.graph_spec.node_sets_spec.keys())

    def _ogb_to_sparse(self,):
        sparse_graphs_csr = dict()
        sparse_graphs_csc = dict()
        for es in self.edge_index_dict.keys():
            source, rel_type, target = es
            edge_list = self.edge_index_dict[es]
            num_source = self.num_nodes_dict[source]
            num_target = self.num_nodes_dict[target]
            data = np.ones(edge_list.shape[1])
            sparse_graphs_csr[rel_type] = csr_matrix((data, (edge_list[0], edge_list[1])), (num_source, num_target))
            sparse_graphs_csc[rel_type] = csc_matrix((data, (edge_list[0], edge_list[1])), (num_source, num_target))
        return sparse_graphs_csr, sparse_graphs_csc

    def _save_node_feat(self,):
        feat_dir = os.path.join(self.output_dir, "node_feat")
        if not os.path.exists(feat_dir):
            os.mkdir(feat_dir)
        for node_set, feature in self.node_feat_dict.items():
            out_path = os.path.join(feat_dir, f"node_feat-{node_set}.csv")
            np.savetxt(out_path, feature, delimiter=",")
            print(f"Saved {node_set} features to {out_path}")
        return

    def save(self,):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        print("Saving Node Features...")
        self._save_node_feat()

    def load(self,):
        return


# custom Dataloader class for OGBN-MAG that includes method
# to create node features for all other node sets
class MAG(Dataloader):
    def __init__(self,
                 ogb_dir: str,
                 output_dir: str,
                 schema_path: str) -> None:
        super().__init__("ogbn-mag", ogb_dir, output_dir, schema_path)
        print("Creating all node features...", end="\r")
        start = time.time()
        self._create_all_node_features()
        elapsed = time.time() - start
        print(f"Created all node features ({elapsed:0.2f} seconds)")

        return

    def _create_node_feature(self, node_set: str, edge_set: str, transpose: bool = False) -> None:
        output, _, input = edge_set
        row, col = self.edge_index_dict[edge_set][0, :], self.edge_index_dict[edge_set][1, :]
        data = np.ones(shape=len(row), dtype=int)
        if transpose:
            row, col = col, row
            input, output = output, input
        input_feature = self.node_feat_dict[input]
        adj = csr_matrix((data, (row, col)))
        num_edges = adj.sum(1)
        # new features are average of features from incident nodes
        output_feature = adj.dot(input_feature) / num_edges
        self.node_feat_dict[output] = output_feature
        return

    def _create_all_node_features(self,) -> None:
        node_sets = ["author", "institution", "field_of_study"]
        edge_sets = [("author", "writes", "paper"),
                     ("author", "affiliated_with", "institution"),
                     ("paper", "has_topic", "field_of_study")]
        transposes = [False, True, True]
        inputs = zip(node_sets, edge_sets, transposes)
        for input in inputs:
            self._create_node_feature(*input)
        return


# just for consistency (same as base Dataloader class)
class Products(Dataloader):
    def __init__(self,
                 ogb_dir: str,
                 output_dir: str,
                 schema_path: str) -> None:
        super().__init__("ogbn-products", ogb_dir, output_dir, schema_path)
        return


class Arxiv(Dataloader):
    def __init__(self,
                 ogb_dir: str,
                 output_dir: str,
                 schema_path: str) -> None:
        super().__init__("ogbn-arxiv", ogb_dir, output_dir, schema_path)
        return


class Papers100M(Dataloader):
    def __init__(self,
                 ogb_dir: str,
                 output_dir: str,
                 schema_path: str) -> None:
        super().__init__("ogbn-papers100M", ogb_dir, output_dir, schema_path)
        return


class MAG240M(Dataloader):
    def __init__(self,
                 ogb_dir: str,
                 output_dir: str,
                 schema_path: str) -> None:
        self.output_dir = output_dir
        graph_schema = tfgnn.read_schema(schema_path)
        self.graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
        print("Loading OGB dataset...", end="\r")
        start = time.time()
        dataset = MAG240MDataset(root=ogb_dir)
        elapsed = time.time() - start
        print(f"Loaded OGB dataset ({elapsed:0.2f} seconds)")

        self.name = "MAG240M"
        self.root = dataset.root

        self.splits = dataset.get_idx_split()
        self.num_classes = dataset.num_classes

        self.labels = {"paper": dataset.paper_label}

        edge_sets = self._get_edge_sets_from_spec()

        self.num_nodes_dict = dict()
        self.num_nodes_dict["paper"] = dataset.num_papers
        self.num_nodes_dict["author"] = dataset.num_authors
        self.num_nodes_dict["institution"] = dataset.num_institutions

        self.node_feat_dict = ()
        self.edge_index_dict = dict()
        for es in edge_sets:
            source, rel, target = es
            self.edge_index_dict[es] = dataset.edge_index(source, rel, target)

        for split in self.splits.keys():
            self.splits[split] = {"paper": self.splits[split]}

        print("Converting to networkx graphs...", end="\r")
        start = time.time()
        self.sparse_graphs_csr, self.sparse_graphs_csc = self._ogb_to_sparse()
        elapsed = time.time() - start
        print(f"Converted to networkx graphs ({elapsed:0.2f} seconds)")

    def _get_edge_sets_from_spec(self,):
        edge_sets = []
        for es in list(self.graph_spec.edge_sets_spec.keys()):
            source_type = self.graph_spec.edge_sets_spec[es].adjacency_spec.source_name
            target_type = self.graph_spec.edge_sets_spec[es].adjacency_spec.target_name
            edge_sets.append((source_type, es, target_type))
        return edge_sets

    def _get_node_sets_from_spec(self,):
        return list(self.graph_spec.node_sets_spec.keys())

    def _ogb_to_sparse(self,):
        sparse_graphs_csr = dict()
        sparse_graphs_csc = dict()
        for es in self.edge_index_dict.keys():
            source, rel_type, target = es
            edge_list = self.edge_index_dict[es]
            num_source = self.num_nodes_dict[source]
            num_target = self.num_nodes_dict[target]
            data = np.ones(edge_list.shape[1])
            sparse_graphs_csr[rel_type] = csr_matrix((data, (edge_list[0], edge_list[1])), (num_source, num_target))
            sparse_graphs_csc[rel_type] = csc_matrix((data, (edge_list[0], edge_list[1])), (num_source, num_target))
        return sparse_graphs_csr, sparse_graphs_csc

    def _save_node_feat(self,):
        feat_dir = os.path.join(self.output_dir, "node_feat")
        if not os.path.exists(feat_dir):
            os.mkdir(feat_dir)
        for node_set, feature in self.node_feat_dict.items():
            out_path = os.path.join(feat_dir, f"node_feat-{node_set}.csv")
            np.savetxt(out_path, feature, delimiter=",")
            print(f"Saved {node_set} features to {out_path}")
        return

    def summary(self,):
        for item in self.num_nodes_dict.items():
            print(item)
        for es, edges in self.edge_index_dict.items():
            print(f"{es}: {edges.shape}")
        print(self.labels["paper"].shape)

    def save(self,):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        print("Saving Node Features...")
        self._save_node_feat()
        print("Saving Graphs...")

    def load(self,):
        return
