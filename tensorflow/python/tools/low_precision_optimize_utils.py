# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np  # NOTE
from tensorflow.python.framework import tensor_util


def get_canonical_tensor_name(name):
    """
    Legal tensor names are like: name, ^name, or name:digits. Please refert to:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/graph/tensor_id.cc#L35
    """
    parts = name.split(":")
    is_control_input = name.startswith("^")
    if len(parts) == 1:
        suffix = "" if is_control_input else ":0"
        return name + suffix
    elif len(parts) == 2 and parts[1].isdecimal() and not is_control_input:
        return name
    else:
        raise Exception("Invalid tensor name: {}".format(name))


def tensor_name_to_node_name(tensor_name):
    return tensor_name.strip().strip("^").split(":")[0]


class SimpleNode:
    def __init__(self, name="", op="", inputs=[], output_nodes=[], tensors={}):
        self.name = name
        self.op = op
        self.inputs = inputs
        # Input tensors.
        self.inputs_tensors = [get_canonical_tensor_name(n) for n in inputs]
        # Output nodes.
        self.output_nodes = output_nodes.copy()
        # Mapping from output tensor name to list of nodes that consume this tensor.
        self.tensors = tensors.copy()

    @property
    def num_inputs(self):
        return len(self.inputs_tensors)

    @property
    def num_outputs(self):
        return len(self.output_nodes)

    @property
    def num_tensors(self):
        return len(self.tensors)

    @property
    def input_nodes(self):
        return [tensor_name_to_node_name(inp) for inp in self.inputs_tensors]

    def __eq__(self, o):
        if not isinstance(o, SimpleNode):
            return False
        return (
            self.name == o.name
            and self.op == o.op
            and self.inputs_tensors == o.inputs_tensors
            and self.output_nodes == o.output_nodes
            and self.tensors == o.tensors
        )

    def __str__(self):
        s = ""
        s += "name          : {}\n".format(self.name)
        s += "op            : {}\n".format(self.op)
        s += "inputs_tensors: {}\n".format(self.inputs_tensors)
        s += "ouput_nodes   : {}\n".format(self.output_nodes)
        s += "tensors       : {}\n".format(self.tensors)
        return s


class SimpleGraph:
    def __init__(self, graph_def):
        self._nodes = [
            SimpleNode(name=n.name, op=n.op, inputs=list(n.input))
            for n in graph_def.node
        ]
        self._name2index = {n.name: i for i, n in enumerate(graph_def.node)}
        self._graph_def = graph_def

        for node in graph_def.node:
            for inp in node.input:
                inp_node_name = tensor_name_to_node_name(inp)
                inp_tensor_name = get_canonical_tensor_name(inp)
                if inp_node_name not in self._name2index:
                    raise Exception(
                        "SimpleNode {}: Unknown input node {}".format(node.name, inp)
                    )
                input_node = self._nodes[self._name2index[inp_node_name]]
                # update input node"s [output_node, ..] list
                input_node.output_nodes.append(node.name)
                # update input node"s {tensor: output_node, ..} dictionary
                #   TODO: we are missing Graph final output node"s tensors,
                #   but it is not possible to inspect how many tensors inside
                #   it, therefore we currently ignore it.
                if inp_tensor_name not in input_node.tensors:
                    input_node.tensors[inp_tensor_name] = [node.name]
                else:
                    input_node.tensors[inp_tensor_name].append(node.name)

    @property
    def num_nodes(self):
        """Get total number of nodes in graph."""
        return len(self._nodes)

    @property
    def nodes(self):
        """Get all nodes in graph."""
        return self._nodes

    def name2index(self, name):
        """Get index of node."""
        if name not in self._name2index:
            error_msg = "Node {} not exists".format(name)
            logging.error(error_msg)
            raise Exception(error_msg)
        return self._name2index[name]

    def node(self, idx):
        """Get node with given index."""
        if idx >= len(self._nodes):
            error_msg = "Node index {} out of range".format(idx)
            logging.error(error_msg)
            raise Exception(error_msg)
        return self._nodes[idx]

    def name2node(self, name):
        """Get node by name."""
        return self.node(self._name2index[name])

    def input_nodes(self, blacklist=["Const"]):
        """Get names of input nodes"""
        return [
            n.name for n in self._nodes if n.num_inputs == 0 and n.op not in blacklist
        ]

    def output_nodes(self):
        """Get names of output nodes, which are those without downstream."""
        return [n.name for n in self._nodes if n.num_outputs == 0]

    def input_nodes_index(self, node_idx):
        """Get indexes of input nodes for node of given index."""
        return [self._name2index[n] for n in self._nodes[node_idx].input_nodes]

    def get_simple_node_by_name(self, name):
        node_name = tensor_name_to_node_name(name)
        if node_name not in self._name2index:
            raise Exception("Unknown node name: {}".format(node_name))
        return self.node(self.name2index(node_name))

    def get_node_by_name(self, name):
        node_name = tensor_name_to_node_name(name)
        if node_name not in self._name2index:
            raise Exception("Unknown node name: {}".format(node_name))
        idx = self._name2index[node_name]
        if idx >= len(self._graph_def.node):
            raise Exception("Unknown node name: {}".format(node_name))
        return self._graph_def.node[idx]


# [Basic graph utils]
def get_const_value(node):
    # Alternatively
    # tf.contrib.util.constant_value(tensor) will get a tensor"s constant value
    return tensor_util.MakeNdarray(node.attr["value"].tensor)


def get_const_value_by_name(graph_def, name, simple_graph=None):
    if simple_graph:
        node = simple_graph.get_node_by_name(name)
        return get_const_value(node)
    else:
        node_name = tensor_name_to_node_name(name)
        founds = [nd for nd in graph_def.node if nd.name == node_name]
        if len(founds) == 0:
            error_msg = "Unknown node name: {}".format(node_name)
            raise Exception(error_msg)
        return get_const_value(founds[0])


# [Pattern matching]
def check_inputs(
    simple_graph, current_node, pattern_nodes, first_node, matched_name_map
):
    # check op type
    if first_node.op != "*":
        matched_ops = [op.strip() for op in first_node.op.split("|")]
        if current_node.op not in matched_ops:
            return False
    # check node name
    if first_node.name in matched_name_map:
        if matched_name_map[first_node.name] != current_node.name:
            return False
    # check inputs
    if (len(first_node.inputs) == 1) and (first_node.inputs[0] == "*"):
        matched_name_map[first_node.name] = current_node.name
        return True
    # if inputs contains both unknown inputs and known inputs
    if (len(first_node.inputs) > 1) and ("*" in first_node.inputs):
        known_inputs = [name for name in first_node.inputs if name != "*"]
        start_idx = 0
        for key_name in known_inputs:
            matched = False
            if key_name.isdigit():
                matched = True
                continue
            for i in range(start_idx, len(current_node.inputs)):
                input_name = current_node.inputs[i]
                cur_input_node = simple_graph.get_simple_node_by_name(input_name)
                expected_input_op_str = (pattern_nodes[key_name].op).strip()
                if "|" in expected_input_op_str:
                    expected_input_ops = expected_input_op_str.split("|")
                else:
                    expected_input_ops = list([expected_input_op_str])
                if (cur_input_node.op in expected_input_ops) and (
                    check_inputs(
                        simple_graph,
                        cur_input_node,
                        pattern_nodes,
                        pattern_nodes[key_name],
                        matched_name_map,
                    )
                ):
                    matched = True
                    start_idx = i
            if not matched:
                return False
    # if all listed inputs are known inputs
    else:
        if len(current_node.inputs) != len(first_node.inputs):
            return False
        for i, input_name in enumerate(current_node.inputs):
            cur_input_node = simple_graph.get_simple_node_by_name(input_name)
            if first_node.inputs[i].isdigit():
                continue
            tmp_input_node = pattern_nodes[first_node.inputs[i]]
            if not check_inputs(
                simple_graph,
                cur_input_node,
                pattern_nodes,
                tmp_input_node,
                matched_name_map,
            ):
                return False
    matched_name_map[first_node.name] = current_node.name
    return True


def get_matched_pattern(
    simple_graph, pattern_nodes, first_node_key, init_name_map=None
):
    matched_name_maps = list()
    for i, node in enumerate(simple_graph.nodes):
        simple_node = simple_graph.get_simple_node_by_name(node.name)
        tmp_name_map = init_name_map.copy() if init_name_map else dict()
        if check_inputs(
            simple_graph,
            simple_node,
            pattern_nodes,
            pattern_nodes[first_node_key],
            tmp_name_map,
        ):
            matched_name_maps.append(tmp_name_map)
    return matched_name_maps


def get_input_target_op_name(simple_graph, node_name, input_index, target_op, op_map):
    node = simple_graph.get_simple_node_by_name(node_name)
    input_node_name = tensor_name_to_node_name(node.inputs[input_index])
    input_node = simple_graph.get_simple_node_by_name(input_node_name)
    if input_node.op == target_op:
        return input_node_name
    if input_node.op in op_map:
        for tmp_index in op_map[input_node.op]:
            target_name = get_input_target_op_name(
                simple_graph, input_node_name, tmp_index, target_op, op_map
            )
            if target_name is not None:
                break
        return target_name
    else:
        return None


def remove_underscore_class_attr(graph_def):
    for i, node in enumerate(graph_def.node):
        if '_class' in node.attr.keys():
            node.attr.pop('_class')


def non_linear_quant_params_search(data, bins=2048, dst_nbins=256):
    min_val, max_val = np.min(data), np.max(data)
    if min_val == max_val:
        if min_val == 0.0:
            return 0.0, 255.0
        return min(0.0, max_val), max(0.0, max_val)
    bin_width = (max_val - min_val) / bins
    hist_data = np.int32(np.minimum(np.floor((data - min_val) / bin_width), bins - 1))
    histogram = np.bincount(hist_data)

    def _get_norm(delta_begin, delta_end, density, norm_type):
        """
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        assert norm_type == "L2", "Only L2 norms are currently supported"
        norm = 0.0
        if norm_type == "L2":
            norm = (
                delta_end * delta_end * delta_end
                - delta_begin * delta_begin * delta_begin
            ) / 3
        return density * norm

    def _compute_quantization_error(next_start_bin, next_end_bin, norm_type):
        """
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = np.arange(bins)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = np.clip(src_bin_begin // dst_bin_width, 0, dst_nbins - 1)
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = np.clip(src_bin_end // dst_bin_width, 0, dst_nbins - 1)
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = histogram / bin_width

        norm = np.zeros(bins)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += _get_norm(delta_begin, delta_end, density, norm_type)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * _get_norm(
            -dst_bin_width / 2, dst_bin_width / 2, density, norm_type
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += _get_norm(delta_begin, delta_end, density, norm_type)

        return np.sum(norm)

    # cumulative sum
    total = sum(histogram)
    cSum = np.cumsum(histogram, axis=0)

    stepsize = 1e-5  # granularity
    alpha = 0.0  # lower bound
    beta = 1.0  # upper bound
    start_bin = 0
    end_bin = bins - 1
    norm_min = float('inf')

    while alpha < beta:
        # Find the next step
        next_alpha = alpha + stepsize
        next_beta = beta - stepsize

        # find the left and right bins between the quantile bounds
        left = start_bin
        right = end_bin
        while left < end_bin and cSum[left] < next_alpha * total:
            left = left + 1
        while right > start_bin and cSum[right] > next_beta * total:
            right = right - 1

        # decide the next move
        next_start_bin = start_bin
        next_end_bin = end_bin
        if (left - start_bin) > (end_bin - right):
            # move the start bin
            next_start_bin = left
            alpha = next_alpha
        else:
            # move the end bin
            next_end_bin = right
            beta = next_beta

        if next_start_bin == start_bin and next_end_bin == end_bin:
            continue

        # calculate the quantization error using next_start_bin and next_end_bin
        norm = _compute_quantization_error(next_start_bin, next_end_bin, "L2")

        if norm > norm_min:
            break
        norm_min = norm
        start_bin = next_start_bin
        end_bin = next_end_bin

    new_min = min_val + bin_width * start_bin
    new_max = min_val + bin_width * (end_bin + 1)

    return new_min, new_max
