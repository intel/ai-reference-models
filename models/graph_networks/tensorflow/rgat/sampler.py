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
import json
from typing import Sequence, Dict, Tuple
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf
import tensorflow_gnn as tfgnn
from dataloader import Dataloader


# base sampling operation class
class SamplingOp():
    def __init__(self, op: Dict):
        self.type = op["op_type"]
        self.name = op["op_name"]
        return

    def summary(self,):
        print(f"OP NAME: {self.name}")
        print(f"OP TYPE: {self.type}")


class OpResult():
    '''
    A class that stores the result of a sampling operation.
    Init Args:
        op_type: One of "seed_op" or "sampling_op"
        op_name: Name of the operation that it will hold the result of
        node_set: The node set name of the returned nodes'
        nodes: A list of sampled nodes IDs
    '''
    def __init__(self, op_type: str, op_name: str, input_node_set: str, output_node_set: str,
                 nodes: Sequence[int], edges: Dict[str, Sequence[Tuple[int, int]]]):
        self.op_type = op_type
        self.op_name = op_name
        self.input_node_set = input_node_set
        self.output_node_set = output_node_set
        self.output_nodes = nodes
        self.edges = edges
        return

    def is_empty(self,):
        return len(self.output_nodes) == 0

    def summary(self,):
        print(f"{self.op_name}:")
        print(f"  input node set: {self.input_node_set}")
        print(f"  output node set: {self.output_node_set}")
        print(f"  sampled {self.output_node_set} nodes: {len(self.output_nodes)}")
        for relation, edges in self.edges.items():
            print(f"  sampled {relation} edges: {len(edges)}")
        return


class Seed(SamplingOp):
    '''
    A seed sampling operation that simply returns the seed node as
    part of an OpResult. This must be the first operation for any
    sampling method.
    Init Args:
        op: A python dictionary representing the seed operation parsed
            from the sampling schema
    '''
    def __init__(self, op: Dict):
        super().__init__(op)
        self.node_set_name = op["node_set_name"]
        self.output_node_set = self.node_set_name

    def execute(self, dataloader: Dataloader, seed: int):
        # print(f"executing {self.name} op")
        return OpResult(self.type, self.name, self.output_node_set, self.output_node_set, [seed], dict())

    def summary(self,):
        print(f"OP NAME: {self.name}")
        print(f"OP TYPE: {self.type}")
        print(f"NODE SET: {self.node_set_name}")
        print(f"OUTPUT NODE SET: {self.output_node_set}\n")


class Sample(SamplingOp):
    '''
    A sampling op that samples a certain number of nodes from a given edge set.
    Init Args:
        op: A python dictionary representing the seed operation parsed
            from the sampling schema
        output_node_set: The name of the node set being sampled.
        output_is_target: True if we are sampling target nodes from a set of
            source nodes. False if we are sampling source nodes from a set of
            target nodes.
    '''
    def __init__(self, op: Dict, output_node_set: str, output_is_target: bool):
        super().__init__(op)
        self.edge_set_name = op["edge_set_name"]
        self.input_ops = op["input_op_names"]
        self.size = op["size"]
        # self.input_node_set = self.input_ops[0].output_node_set
        self.output_is_target = output_is_target
        self.output_node_set = output_node_set

    def execute(self, dataloader: Dataloader, input: Sequence[OpResult]):
        input_nodes = self._aggregate_inputs(input)
        edges_sampled = set()
        nodes_sampled = set()
        if len(input_nodes) != 0:
            for node in input_nodes:
                if self.output_is_target:
                    # n = list(graph.successors(node))
                    # g = dataloader.nx_graphs[0]
                    graph = dataloader.sparse_graphs_csr[self.edge_set_name]
                    _, n = graph.getrow(node).nonzero()
                else:
                    # n = list(graph.predecessors(node))
                    # g = dataloader.nx_graphs[1]
                    graph = dataloader.sparse_graphs_csc[self.edge_set_name]
                    n, _ = graph.getcol(node).nonzero()
                # n = list(nx.all_neighbors(graph, node))
                # TODO: make sure edge sets with one node type can sample all neighbors
                n = list(n)
                if len(n) > self.size:
                    n = random.sample(n, self.size)
                nodes_sampled.update(n)
                edges_sampled.update([(node, i) for i in n])
            nodes_sampled = list(nodes_sampled)
            edges_sampled = {self.edge_set_name: list(edges_sampled)}
        return OpResult(self.type, self.name, input[0].output_node_set,
                        self.output_node_set, nodes_sampled, edges_sampled)

    def _aggregate_inputs(self, input: Sequence[OpResult]):
        output_node_set = input[0].output_node_set
        new_nodes = set()
        for result in input:
            assert (result.output_node_set == output_node_set)
            new_nodes.update(result.output_nodes)

        new_nodes = list(new_nodes)
        return new_nodes

    def summary(self,):
        print(f"OP NAME: {self.name}")
        print(f"OP TYPE: {self.type}")
        print(f"EDGE SET: {self.edge_set_name}")
        print(f"INPUT OPS: {self.input_ops}")
        print(f"OUTPUT SIZE: {self.size}")
        print(f"OUTPUT NODE SET: {self.output_node_set}\n")


class SamplingRoutine():
    '''
    The SamplingRoutine runs the sampling scheme for a single root node.
    It is meant to be used by the Sampler class which will run it on a list
    of root nodes.
    Init Args:
        sampling_schema_path: Path to the json file representing the sampling
            schema.
        graph_spec: A TFGNN GraphTensorSpec that defines the final subgraph.
    '''
    def __init__(self, sampling_schema_path: str, graph_spec: tfgnn.GraphTensorSpec):
        self.graph_spec = graph_spec
        self.op_order = []
        self.op_dict = dict()
        # TODO: method that does everything below and further validates sampling schema
        op_list = json.load(open(sampling_schema_path, 'r'))
        seed_op = op_list.pop(0)
        assert (seed_op["op_type"] == "seed_op")
        self.op_order.append(seed_op["op_name"])
        self.op_dict[seed_op["op_name"]] = Seed(seed_op)
        for _op in op_list:
            es = _op["edge_set_name"]
            source_node_set = self.graph_spec.edge_sets_spec[es].adjacency_spec.source_name
            target_node_set = self.graph_spec.edge_sets_spec[es].adjacency_spec.target_name
            input_node_set = self.op_dict[_op["input_op_names"][0]].output_node_set
            if input_node_set == source_node_set:
                output_node_set = target_node_set
                output_is_target = True
            else:
                output_node_set = source_node_set
                output_is_target = False
            op = Sample(_op, output_node_set, output_is_target)
            self.op_order.append(op.name)
            self.op_dict[op.name] = op
        self.summary()
        return

    def summary(self,):
        print("\nSAMPLING ROUTINE:\n")
        for op_name in self.op_order:
            op = self.op_dict[op_name]
            op.summary()

    def run(self, dataloader: Dataloader, seed: int):
        seed_op = self.op_dict[self.op_order[0]]
        op_results = dict()
        op_results[seed_op.name] = seed_op.execute(dataloader, seed)
        for op_name in self.op_order[1:]:
            op = self.op_dict[op_name]
            input = [op_results[io] for io in op.input_ops]
            op_results[op.name] = op.execute(dataloader, input)
            # op_results[op.name].summary()
        output = self._aggregate_output(op_results)
        return output

    def _aggregate_output(self, op_results):
        node_sets = list(self.graph_spec.node_sets_spec.keys())
        edge_sets = list(self.graph_spec.edge_sets_spec.keys())
        node_output = {ns: set() for ns in node_sets}
        edge_output = {es: set() for es in edge_sets}
        for op_name, result in op_results.items():
            if result.is_empty():
                continue
            node_output[result.output_node_set].update(result.output_nodes)
            for relation, edges in result.edges.items():
                edge_output[relation].update(edges)
        for ns in node_output.keys():
            node_output[ns] = list(node_output[ns])
        for es in edge_output.keys():
            edge_output[es] = list(edge_output[es])
        return node_output, edge_output


class Sampler():
    '''
    The Sampler class creates a sampling routine and runs it on a seqeuence
    of supplied root nodes. It then creates TFGNN GraphTensors out of the
    sampling output and saves it to a sharded TFRecord dataset.
    Init Args:
        dataloader: Dataloader instance to sample from.
        sampling_schema_path: Path to a json file representing the sampling
            schema.
    '''
    def __init__(self,
                 dataloader: Dataloader,
                 sampling_schema_path: str):
        print("dataloader: ", dataloader, sampling_schema_path)
        self.dataloader = dataloader
        self.sampling_schema_path = sampling_schema_path
        self.sampling_routine = SamplingRoutine(self.sampling_schema_path, self.dataloader.graph_spec)
        self.target = self.sampling_routine.op_dict[self.sampling_routine.op_order[0]].node_set_name
        return

    def _get_node_set_maps(self, sampled_node_sets):
        node_maps = dict()
        for ns, nodes in sampled_node_sets.items():
            node_maps[ns] = {nodes[i]: i for i in range(len(nodes))}
        return node_maps

    def _create_tfgnn_node_sets(self, node_set, nodes):
        features = dict()
        features["index"] = np.array(nodes).astype(np.int64)
        return tfgnn.NodeSet.from_fields(sizes=[len(nodes)], features=features)

    def _create_tfgnn_edge_sets(self, edge_set, edges, node_maps):
        source_type = self.dataloader.graph_spec.edge_sets_spec[edge_set].adjacency_spec.source_name
        target_type = self.dataloader.graph_spec.edge_sets_spec[edge_set].adjacency_spec.target_name
        source_map = node_maps[source_type]
        target_map = node_maps[target_type]
        edges = [(source_map[s], target_map[t]) for (s, t) in edges if ((s in source_map) and (t in target_map))]
        num_edges = len(edges)
        if num_edges > 0:
            source, target = zip(*edges)
            adjacency = tfgnn.Adjacency.from_indices(
                (source_type, source),
                (target_type, target))
            return tfgnn.EdgeSet.from_fields(
                sizes=[num_edges],
                adjacency=adjacency)
        return

    # Given a single root node, samples a subgraph and returns it as a graph tensor
    def sample_subgraph(self, seed: int):
        tfgnn_node_sets = dict()
        tfgnn_edge_sets = dict()
        # sample nodes
        sampled_nodes, sampled_edges = self.sampling_routine.run(self.dataloader, seed)
        # create tfgnn context
        label = self.dataloader.labels[self.target][seed]
        context_feat = {"index": [seed]}
        if not np.isnan(label):
            context_feat["label"] = [int(label)]
        tfgnn_context = tfgnn.Context.from_fields(
            features=context_feat)
        # create tfgnn node sets
        for ns, nodes in sampled_nodes.items():
            tfgnn_node_sets[ns] = self._create_tfgnn_node_sets(ns, nodes)
        # create tfgnn edge sets
        edge_sets = list(self.dataloader.graph_spec.edge_sets_spec.keys())
        node_maps = self._get_node_set_maps(sampled_nodes)
        for es in edge_sets:
            tfgnn_es = self._create_tfgnn_edge_sets(es, sampled_edges[es], node_maps)
            if tfgnn_es:
                tfgnn_edge_sets[es] = tfgnn_es
        # create graph tensor
        gt = tfgnn.GraphTensor.from_pieces(
            node_sets=tfgnn_node_sets,
            edge_sets=tfgnn_edge_sets,
            context=tfgnn_context)
        return gt

    # samples a number of subgraphs and saves them to a single TFRecord file
    def write_shard(self, seeds: Sequence[int], shard_path: str) -> None:
        with tf.io.TFRecordWriter(shard_path) as writer:
            for seed in seeds:
                gt = self.sample_subgraph(seed)
                example = tfgnn.write_example(gt)
                writer.write(example.SerializeToString())
        return

    # Runs sampling for all root nodes and saves them to multiple TFRecord files
    def sample_dataset(self, split: str, num_workers: int = 1, dir: str = None) -> None:
        if dir is None:
            dir = self.dataloader.root
        dir = os.path.join(dir, split)
        if not os.path.exists(dir):
            os.makedirs(dir)
        split_idx = self.dataloader.splits[split][self.target]
        num_nodes = split_idx.shape[0]
        index = 0
        n_graphs_shard = 1000
        n_shards = int(num_nodes / n_graphs_shard) + (1 if num_nodes % n_graphs_shard != 0 else 0)
        for shard in tqdm(range(n_shards), disable=False):
            shard_file = "{}_{}_{}.records".format(self.dataloader.name, split, '%.5d-of-%.5d' % (shard, n_shards - 1))
            shard_path = os.path.join(dir, shard_file)
            end = index + n_graphs_shard if num_nodes > (index + n_graphs_shard) else -1
            nodes_shard_list = split_idx[index:end]
            self.write_shard(nodes_shard_list, shard_path)
            index = end
        return
