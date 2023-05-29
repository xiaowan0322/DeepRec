import os
import tempfile

import numpy as np
from tensorflow.core.framework.attr_value_pb2 import AttrValue
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import (
    dtypes,
    graph_util,
    meta_graph,
    ops,
    versions,
)
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import (
    array_ops,
    gen_array_ops,
    gen_kv_variable_ops,
    gen_math_ops,
    init_ops,
    math_ops,
    nn_ops,
)
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.variable_scope import get_embedding_variable, get_variable
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader, loader_impl
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_utils import list_variables
from tensorflow.python.util import quantize_embedding_variable

import low_precision_optimize_utils as util  # NOTE
from low_precision_optimize_utils import SimpleGraph, SimpleNode  # NOTE

INT8 = 'INT8'
BF16 = 'BF16'
FP16 = 'FP16'

ev_attrs = [
    '_block_num',
    '_counter_type',
    '_default_value_dim',
    '_emb_index',
    '_false_positive_probability',
    '_filter_freq',
    '_ht_partition_num',
    '_ht_type',
    '_init_data_source',
    '_invalid_key_type',
    '_is_sparse',
    '_l2_weight_threshold',
    '_layout',
    '_max_element_size',
    '_save_slice_info',
    '_slot_index',
    '_slot_num',
    '_steps_to_live',
    '_storage_path',
    '_storage_size',
    '_storage_type',
]

ev_suffix = [
    'freqs',
    'freqs_filtered',
    'keys',
    'keys_filtered',
    'partition_filter_offset',
    'partition_offset',
    'values',
    'versions',
    'versions_filtered',
]


def _ts(name):
    return util.get_canonical_tensor_name(name)


def _nd(name):
    return util.tensor_name_to_node_name(name)


def get_all_variables():
    all_variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    all_variables += ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS)
    return list(set(all_variables))


def get_variable_by_name(name):
    all_variables = get_all_variables()
    for var in all_variables:
        if var.name == name:
            return var
    return None


def get_tf_dtype(dtype):
    if dtype == INT8:
        return dtypes.int8
    elif dtype in [BF16, FP16]:
        return dtypes.bfloat16 if dtype == BF16 else dtypes.float16
    else:
        raise Exception('Unsupported data type: {}'.format(dtype))


def update_op_inputs(graph, rename_dict):
    rename_dict = {_ts(k): v for k, v in rename_dict.items()}
    src_names = list(rename_dict.keys())
    for op in graph.get_operations():
        for i, tensor in enumerate(op.inputs):
            if tensor.name in src_names:
                op._update_input(i, rename_dict[tensor.name])


def create_new_meta_graph_def(graph, meta_graph_def, new_graph_def, new_saver=None):
    new_nodes = [node.name for node in new_graph_def.node]
    old_nodes = [node.name for node in graph.as_graph_def().node]
    removed_nodes = list(set(old_nodes).difference(set(new_nodes)))
    new_mgd = meta_graph_pb2.MetaGraphDef()
    # Add meta info def
    meta_info_def = meta_graph_pb2.MetaGraphDef.MetaInfoDef()
    meta_info_def.tensorflow_version = versions.__version__
    meta_info_def.tensorflow_git_version = versions.__git_version__
    stripped_op_list = meta_graph.stripped_op_list_for_graph(new_graph_def)
    meta_info_def.stripped_op_list.MergeFrom(stripped_op_list)
    new_mgd.meta_info_def.MergeFrom(meta_info_def)
    # Add graph def
    new_mgd.graph_def.MergeFrom(new_graph_def)
    # Add saver def
    new_mgd.saver_def.MergeFrom(new_saver.saver_def)
    # Add collection list
    clist = graph.get_all_collection_keys()
    exclude_nodes = [_ts(name) for name in removed_nodes] + removed_nodes
    for ctype in clist:
        if ctype in [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.TRAINABLE_VARIABLES]:
            meta_graph.add_collection_def(
                new_mgd,
                ctype,
                graph=graph,
                export_scope=None,
                exclude_nodes=exclude_nodes,
            )
            values = new_mgd.collection_def[ctype].bytes_list.value
            new_mgd.collection_def[ctype].bytes_list.Clear()
            for value in values:
                proto = ops.get_collection_proto_type(ctype)()
                proto.ParseFromString(value)
                if proto.variable_name in exclude_nodes:
                    continue
                for attr in ['initial_value_name', 'snapshot_name', 'initializer_name']:
                    if getattr(proto, attr) in exclude_nodes:
                        setattr(proto, attr, proto.variable_name)
                new_value = proto.SerializeToString()
                new_mgd.collection_def[ctype].bytes_list.value.append(new_value)

    for tag in meta_graph_def.meta_info_def.tags:
        new_mgd.meta_info_def.tags.append(tag)
    for key in meta_graph_def.signature_def:
        new_mgd.signature_def[key].CopyFrom(meta_graph_def.signature_def[key])
    return new_mgd


def remove_redundant_quants(session, graph_def):
    simple_graph = SimpleGraph(graph_def)

    def _get_pattern():
        pl = list()
        pl.append(SimpleNode('quantize', 'QuantizeV2', ['dequantize', '0', '1'], ['0']))
        pl.append(SimpleNode('dequantize', 'Dequantize', ['2', '3', '4'], ['quantize']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    pattern, first_key = _get_pattern()
    ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
    for ptm in ptm_list:
        dequantize_op = session.graph.get_operation_by_name(ptm['dequantize'])
        update_op_inputs(session.graph, {ptm['quantize']: dequantize_op.inputs[0]})


def remove_redundant_casts(session, graph_def):
    simple_graph = SimpleGraph(graph_def)

    def _get_pattern():
        pl = list()
        pl.append(SimpleNode('cast2src', 'Cast', ['cast2dst'], ['0']))
        pl.append(SimpleNode('cast2dst', 'Cast', ['0'], ['cast2src']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    pattern, first_key = _get_pattern()
    ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
    for ptm in ptm_list:
        cast2src_op = session.graph.get_operation_by_name(ptm['cast2src'])
        cast2dst_op = session.graph.get_operation_by_name(ptm['cast2dst'])
        if cast2dst_op.get_attr('SrcT') == cast2src_op.get_attr('DstT'):
            update_op_inputs(session.graph, {ptm['cast2src']: cast2dst_op.inputs[0]})


def dense_opt(session, graph_def, opt_config, data_type, calib_file):
    simple_graph = SimpleGraph(graph_def)
    update_dict = dict()
    calib_data = None
    if calib_file:
        calib_data = np.load(calib_file, allow_pickle=True, encoding='bytes')

    def _calibrate(ts_name):
        assert calib_data is not None, 'Calibration data needed for INT8 optimization.'
        values = [session.run(ts_name, feed_dict=fd) for fd in calib_data]
        values = np.concatenate([v.ravel() for v in values])
        return util.non_linear_quant_params_search(values)

    def _get_matmul_pattern(with_bias, with_relu):
        pl = list()
        output = ['0']
        if with_relu:
            tmp_input = ['bias_add'] if with_bias else ['matmul']
            pl.append(SimpleNode('relu', 'Relu', tmp_input, ['1']))
            output = ['relu']
        if with_bias:
            pl.append(SimpleNode('bias_add', 'BiasAdd', ['matmul', '2'], output))
            output = ['bias_add']
        pl.append(SimpleNode('matmul', 'MatMul', ['0', '1'], output))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    def _get_weight_data(node_name, input_index=1):
        weight_name = util.get_input_target_op_name(
            simple_graph, node_name, input_index, 'Const', {'Identity': [0]}
        )
        if weight_name:
            data = util.get_const_value_by_name(graph_def, weight_name, simple_graph)
        else:
            node = simple_graph.get_node_by_name(node_name)
            try:
                data = session.run(_ts(node.input[input_index]))
            except Exception:
                return None
        return data

    def _optimize(with_bias, with_relu):
        pattern, first_key = _get_matmul_pattern(with_bias, with_relu)
        ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
        ptm_list = [ptm for ptm in ptm_list if ptm['matmul'] not in update_dict]
        if opt_config:
            ptm_list = [ptm for ptm in ptm_list if ptm['matmul'] in opt_config]
        for ptm in ptm_list:
            if ptm['matmul'] in update_dict:
                continue
            w_data = _get_weight_data(ptm['matmul'])
            if with_bias:
                bias_data = _get_weight_data(ptm['bias_add'])
            if w_data is None or (with_bias and bias_data is None):
                continue
            node = simple_graph.get_node_by_name(ptm['matmul'])
            opt_dtype = opt_config.get(node.name) if opt_config else data_type
            print('Optimize dense op to {}: {}'.format(opt_dtype, node.name))
            update_dict[node.name] = [opt_dtype]
            dense_op = session.graph.get_operation_by_name(node.name)
            if opt_dtype in [BF16, FP16]:
                tf_dtype = dtypes.bfloat16 if opt_dtype == BF16 else dtypes.float16
                prefix = '{}/{}'.format(node.name, opt_dtype.lower())
                w_f16_ts = constant(
                    value=math_ops.cast(w_data, tf_dtype).eval(),
                    dtype=tf_dtype,
                    name='{}_weight'.format(prefix),
                )
                in_f16_ts = math_ops.cast(
                    dense_op.inputs[0], tf_dtype, name='{}_input'.format(prefix)
                )
                out_f16_ts = math_ops.matmul(
                    a=in_f16_ts,
                    b=w_f16_ts,
                    transpose_a=dense_op.get_attr('transpose_a'),
                    transpose_b=dense_op.get_attr('transpose_b'),
                    name='{}_matmul'.format(prefix),
                )
                if with_bias:
                    bias_f16_ts = constant(
                        value=math_ops.cast(bias_data, tf_dtype).eval(),
                        dtype=tf_dtype,
                        name='{}_bias'.format(prefix),
                    )
                    out_f16_ts = nn_ops.bias_add(out_f16_ts, bias_f16_ts)
                out_f16_ts = nn_ops.relu(out_f16_ts) if with_relu else out_f16_ts
                out_fp32_ts = math_ops.cast(
                    out_f16_ts, dtypes.float32, name='{}/cast_to_fp32'.format(node.name)
                )
                update_op_inputs(session.graph, {ptm[first_key]: out_fp32_ts})
                continue
            elif opt_dtype != INT8:
                raise Exception('Unsupported data type: {}'.format(opt_dtype))

            def _name(name, prefix=node.name):
                return '{}/{}'.format(prefix, name)

            # Optimize to INT8
            # Update weight
            w_max_abs_val = np.max(np.abs(w_data))
            w_scale = np.array(w_max_abs_val / 127.0)
            w_int8_data = np.int8(np.round(w_data / w_scale))
            w_min_ts = constant(-1 * w_max_abs_val, dtypes.float32, name=_name('w_min'))
            w_max_ts = constant(w_max_abs_val, dtypes.float32, name=_name('w_max'))
            w_int8_ts = constant(w_int8_data, dtypes.qint8, name=_name('int8_weight'))
            # Update input
            in_min_val, in_max_val = _calibrate(_ts(node.input[0]))
            in_min_ts = constant(in_min_val, dtypes.float32, name=_name('in_min'))
            in_max_ts = constant(in_max_val, dtypes.float32, name=_name('in_max'))
            in_int8_ts, _, _ = gen_array_ops.quantize_v2(
                input=dense_op.inputs[0],
                min_range=in_min_ts,
                max_range=in_max_ts,
                T=dtypes.quint8,
                mode='MIN_FIRST',
                name=_name('int8_input'),
            )
            # Add requantize scale
            out_min_val, out_max_val = _calibrate(_ts(ptm[first_key]))
            req_min_val, req_max_val = 0.0, (out_max_val - out_min_val) * 256.0 / 255.0
            req_min_ts = constant(req_min_val, dtypes.float32, name=_name('req_min'))
            req_max_ts = constant(req_max_val, dtypes.float32, name=_name('req_max'))
            # Update bias
            in_scale = (in_max_val - in_min_val) / 255.0
            in_zero_point = -1.0 * round(in_min_val / in_scale)
            compensation = np.sum(-1.0 * in_zero_point * w_int8_data, 0)
            out_scale = (out_max_val - out_min_val) / 255.0
            out_zero_point = -1.0 * round(out_min_val / out_scale)
            deq_scale = w_scale * in_scale
            compensation += out_zero_point * out_scale / deq_scale
            if with_bias:
                bias_int_data = np.int32(compensation + bias_data / deq_scale)
            else:
                bias_int_data = np.int32(compensation)
            bias_int_ts = constant(bias_int_data, dtypes.qint32, name=_name('int_bias'))
            # Update MatMul
            quant_matmul = (
                gen_array_ops.quantized_matmul_with_bias_and_relu_and_requantize
                if with_relu
                else gen_array_ops.quantized_matmul_with_bias_and_requantize
            )
            matmul_int8_ts, _, _ = quant_matmul(
                a=in_int8_ts,
                b=w_int8_ts,
                bias=bias_int_ts,
                min_a=in_min_ts,
                max_a=in_max_ts,
                min_b=w_min_ts,
                max_b=w_max_ts,
                min_freezed_output=req_min_ts,
                max_freezed_output=req_max_ts,
                Toutput=dtypes.quint8,
                input_quant_mode='MIN_FIRST',
                name=_name('int8_matmul'),
            )
            # Add dequantize
            out_min_ts = constant(out_min_val, dtypes.float32, name=_name('out_min'))
            out_max_ts = constant(out_max_val, dtypes.float32, name=_name('out_max'))
            out_fp32_ts = gen_array_ops.dequantize(
                input=matmul_int8_ts,
                min_range=out_min_ts,
                max_range=out_max_ts,
                mode='MIN_FIRST',
                name=_name('dequantize'),
            )
            update_op_inputs(session.graph, {ptm[first_key]: out_fp32_ts})

    for with_relu in [True, False]:
        for with_bias in [True, False]:
            _optimize(with_bias, with_relu)

    remove_redundant_quants(session, session.graph.as_graph_def())
    remove_redundant_casts(session, session.graph.as_graph_def())

    return update_dict


def update_embedding_vars(session):
    update_dict = dict()
    node_dic = {nd.name: nd for nd in session.graph_def.node}
    for op in session.graph.get_operations():
        if op.type == 'KvResourceImportV2':
            var = get_variable_by_name(op.inputs[1].name)
            var._initial_value = op.inputs[3]
            name = _nd(op.inputs[5].name)
            var._invalid_key = util.get_const_value(node_dic[name])
            update_dict[var.name] = op.name

    return update_dict


def embedding_opt(session, graph_def, opt_config, data_type, variable_path):
    simple_graph = SimpleGraph(graph_def)

    def _get_gather_pattern():
        pl = list()
        pl.append(SimpleNode('gather', 'GatherV2', ['read', '0', '1'], ['0']))
        pl.append(SimpleNode('read', 'Identity', ['embed'], ['gather']))
        pl.append(SimpleNode('embed', 'Const', [], ['read']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    update_dict = dict()
    pattern, first_key = _get_gather_pattern()
    ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
    if opt_config:
        ptm_list = [ptm for ptm in ptm_list if ptm['embed'] in opt_config]
    for ptm in ptm_list:
        embed_node = simple_graph.get_node_by_name(ptm['embed'])
        opt_dtype = opt_config.get(embed_node.name) if opt_config else data_type
        if embed_node.name not in update_dict:
            print('Optimize embedding to {}: {}'.format(opt_dtype, embed_node.name))
            # Add variables
            fp32_data = util.get_const_value_by_name(
                graph_def, ptm['embed'], simple_graph
            )
            if opt_dtype == INT8:
                int8_name = '{}/int8_data'.format(embed_node.name)
                int8_var = get_variable(int8_name, fp32_data.shape, dtypes.int8)
                scale_name = '{}/int8_scale'.format(embed_node.name)
                scale_shape = int8_var.shape[-1:]
                scale_var = get_variable(scale_name, scale_shape, dtypes.float32)
                update_dict[embed_node.name] = [int8_var, scale_var, opt_dtype]
            elif opt_dtype in [BF16, FP16]:
                tf_dtype = dtypes.bfloat16 if opt_dtype == BF16 else dtypes.float16
                f16_name = '{}/{}_data'.format(embed_node.name, opt_dtype.lower())
                f16_var = get_variable(f16_name, fp32_data.shape, tf_dtype)
                update_dict[embed_node.name] = [f16_var, opt_dtype]
            else:
                raise Exception('Unsupported data type: {}'.format(opt_dtype))
        # Update Graph
        gather_op = session.graph.get_operation_by_name(ptm['gather'])
        opt_gather = array_ops.gather(
            params=update_dict[embed_node.name][0],
            indices=gather_op.inputs[1],
            axis=gather_op.inputs[2],
            batch_dims=gather_op.get_attr('batch_dims'),
            name='{}/{}'.format(ptm['gather'], opt_dtype.lower()),
        )
        cast_name = '{}/cast_to_fp32'.format(ptm['gather'])
        update_tensor = math_ops.cast(opt_gather, dtype=dtypes.float32, name=cast_name)
        if opt_dtype == INT8:
            rescale_name = '{}/rescale'.format(ptm['gather'])
            scale_var = update_dict[embed_node.name][1]
            update_tensor = math_ops.multiply(update_tensor, scale_var, rescale_name)
        update_op_inputs(session.graph, {ptm['gather']: update_tensor})

    # Convert checkpoint
    for opt_dtype in [INT8, BF16, FP16]:
        opt_dict = {k: v for k, v in update_dict.items() if v[-1] == opt_dtype}
        if len(opt_dict) > 0:
            names = [_nd(key) for key in opt_dict.keys()]
            quant_vars = [opt_dict[name][0] for name in names]
            quant_names = [_nd(var.name) for var in quant_vars]
            if opt_dtype == INT8:
                scale_names = [_nd(opt_dict[name][1].name) for name in names]
                scale_vars = [opt_dict[name][1] for name in names]
            else:
                scale_names, scale_vars = [], []
            tmp_path = tempfile.mkdtemp(dir='.')
            opt_path = '{}/variables'.format(tmp_path)
            dtype = get_tf_dtype(opt_dtype)
            quantize_embedding_variable.quantize_by_name(
                variable_path, opt_path, names, quant_names, scale_names, dtype, False
            )
            saver = saver_lib.Saver(quant_vars + scale_vars)
            saver.restore(session, opt_path)
            gfile.DeleteRecursively(tmp_path)

    return update_dict


def embedding_var_opt(session, graph_def, opt_config, data_type, variable_path):
    simple_graph = SimpleGraph(graph_def)

    def _get_gather_pattern():
        pl = list()
        pl.append(SimpleNode('gather', 'KvResourceGather', ['embed', '0', '1'], ['0']))
        pl.append(SimpleNode('embed', 'KvVarHandleOp', [], ['gather']))
        pattern_nodes = {node.name: node for node in pl}
        return pattern_nodes, pl[0].name

    update_dict = dict()
    pattern, first_key = _get_gather_pattern()
    ptm_list = util.get_matched_pattern(simple_graph, pattern, first_key)
    if opt_config:
        ptm_list = [ptm for ptm in ptm_list if ptm['embed'] in opt_config]
    for ptm in ptm_list:
        embed_name = ptm['embed']
        opt_dtype = opt_config.get(embed_name) if opt_config else data_type
        if embed_name not in update_dict:
            embed_node = util.get_simple_node_by_name(simple_graph, embed_name)
            if embed_node.output_nodes != [ptm['gather']]:
                continue
            print('Optimize embedding variable to {}: {}'.format(opt_dtype, embed_name))
            # Add variables
            var = get_variable_by_name(_ts(embed_name))
            embedding_dim = var.shape[0].value
            value_dtype = get_tf_dtype(opt_dtype)
            opt_var = get_embedding_variable(
                name='{}/{}_data'.format(embed_name, opt_dtype.lower()),
                embedding_dim=embedding_dim,
                key_dtype=var._invalid_key_type,
                value_dtype=value_dtype,
                initializer=init_ops.zeros_initializer(value_dtype),
            )
            for attr in ev_attrs:
                setattr(opt_var, attr, getattr(var, attr))
            if opt_dtype == INT8:
                scale_name = '{}/int8_scale'.format(embed_name)
                scale_var = get_variable(scale_name, [embedding_dim], dtypes.float32)
                update_dict[embed_name] = [opt_var, scale_var, opt_dtype]
            elif opt_dtype in [BF16, FP16]:
                update_dict[embed_name] = [opt_var, opt_dtype]

        # Update Graph
        gather_op = session.graph.get_operation_by_name(ptm['gather'])
        use_default = gather_op.get_attr('is_use_default_value_tensor')
        if use_default:
            init_val = gather_op.inputs[2]
            if opt_dtype == INT8:
                init_val = math_ops.divide(init_val, update_dict[embed_name][1])
                init_val = gen_math_ops.clip_by_value(init_val, -128, 127)
            init_val = math_ops.cast(init_val, value_dtype)
        else:
            init_val = constant(1, dtype=value_dtype)
        opt_gather = gen_kv_variable_ops.kv_resource_gather(
            resource=update_dict[embed_name][0]._handle,
            indices=gather_op.inputs[1],
            default_value=init_val,
            is_use_default_value_tensor=use_default,
            name='{}/{}'.format(ptm['gather'], opt_dtype.lower()),
        )
        cast_name = '{}/cast_to_fp32'.format(ptm['gather'])
        update_tensor = math_ops.cast(opt_gather, dtype=dtypes.float32, name=cast_name)
        if opt_dtype == INT8:
            rescale_name = '{}/rescale'.format(ptm['gather'])
            scale_var = update_dict[embed_name][1]
            update_tensor = math_ops.multiply(update_tensor, scale_var, rescale_name)
        update_op_inputs(session.graph, {ptm['gather']: update_tensor})

    if len(update_dict) == 0:
        return update_dict, None

    # Convert checkpoint
    input_path = variable_path
    for opt_dtype in [INT8, BF16, FP16]:
        opt_dict = {k: v for k, v in update_dict.items() if v[-1] == opt_dtype}
        if len(opt_dict) > 0:
            names = [_nd(key) for key in opt_dict.keys()]
            quant_names = [_nd(opt_dict[name][0].name) for name in names]
            if opt_dtype == INT8:
                scale_names = [_nd(opt_dict[name][1].name) for name in names]
                scale_variables = [opt_dict[name][1] for name in names]
                session.run(variables_lib.variables_initializer(scale_variables))
            else:
                scale_names = []
            tmp_path = tempfile.mkdtemp(dir='.')
            opt_path = '{}/variables'.format(tmp_path)
            dtype = get_tf_dtype(opt_dtype)
            quantize_embedding_variable.quantize_by_name(
                variable_path, opt_path, names, quant_names, scale_names, dtype, True
            )
            if variable_path != input_path:
                gfile.DeleteRecursively(os.path.dirname(variable_path))
            variable_path = opt_path
    return update_dict, variable_path


def optimize(model_path, save_path, opt_config=None, data_type=BF16, calib_file=None):
    saved_model = loader_impl._parse_saved_model(model_path)
    tags = saved_model.meta_graphs[0].meta_info_def.tags
    with session_lib.Session() as sess:
        meta_graph_def = loader.load(sess, tags, model_path)
        signature_keys = list(meta_graph_def.signature_def.keys())
        signature_def = meta_graph_def.signature_def[signature_keys[0]]
        model_outputs = [_nd(v.name) for v in signature_def.outputs.values()]
        init_op = loader_impl.get_init_op(meta_graph_def)
        if init_op is not None:
            model_outputs.append(init_op.name)
        frozen_gdef = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, model_outputs
        )

        # Embedding & Dense optimization
        dense_opt_dict = dense_opt(sess, frozen_gdef, opt_config, data_type, calib_file)
        variable_path = os.path.join(model_path, 'variables/variables')
        embed_opt_dict = embedding_opt(
            sess, frozen_gdef, opt_config, data_type, variable_path
        )
        ev_dict = update_embedding_vars(sess)
        if len(ev_dict) > 0:
            model_outputs.append(_nd(training_util.get_global_step().name))

        def _extract_sub_graph(outputs):
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            util.remove_underscore_class_attr(graph_def)
            return graph_util.extract_sub_graph(graph_def, outputs)

        def _save(save_path):
            sub_graph_def = _extract_sub_graph(model_outputs)
            node_names = [node.name for node in sub_graph_def.node]
            variables = [v for v in get_all_variables() if _nd(v.name) in node_names]
            init_name = variables_lib.variables_initializer(variables).name
            saver = saver_lib.Saver(variables, sharded=True, allow_empty=True)
            saver.save(sess, save_path, write_meta_graph=False, write_state=False)
            return saver, init_name

        # Create Saver
        tmp_path = tempfile.mkdtemp(dir='.')
        variable_path = '{}/variables'.format(tmp_path)
        saver, init_name = _save(variable_path)
        # Optimize embedding variables
        ev_opt_dict, opt_variable_path = embedding_var_opt(
            sess, frozen_gdef, opt_config, data_type, variable_path
        )
        if len(ev_opt_dict) > 0:
            saver, init_name = _save(variable_path)
            variable_path = opt_variable_path

        saver_nodes = [
            saver.saver_def.restore_op_name,
            _nd(saver.saver_def.filename_tensor_name),
            _nd(saver.saver_def.save_tensor_name),
        ]
        graph_def = _extract_sub_graph(model_outputs + saver_nodes + [init_name])
        graph = sess.graph

    # Create new meta graph def
    new_mgd = create_new_meta_graph_def(graph, meta_graph_def, graph_def, saver)

    # Export saved_model
    ops.reset_default_graph()
    with session_lib.Session(graph=ops.Graph()) as sess:
        meta_graph.import_scoped_meta_graph(new_mgd)
        restore_feed_dict = {new_mgd.saver_def.filename_tensor_name: variable_path}
        sess.run(new_mgd.saver_def.restore_op_name, restore_feed_dict)
        # Update embedding variables
        update_embedding_vars(sess)
        # Update assets file
        assets_collection = None
        asset_dict = loader_impl.get_asset_tensors(model_path, meta_graph_def)
        if asset_dict is not None:
            for tensor_name, filename in asset_dict.items():
                node_name = _nd(tensor_name)
                if node_name in [nd.name for nd in sess.graph_def.node]:
                    asset_op = sess.graph.get_operation_by_name(node_name)
                    ts_proto = TensorProto(
                        tensor_shape=asset_op.get_attr('value').tensor_shape,
                        dtype=asset_op.get_attr('value').dtype,
                        string_val=[filename],
                    )
                    asset_op._set_attr('value', AttrValue(tensor=ts_proto))
                    asset_name = asset_op.outputs[0]
                else:
                    asset_name = constant(filename, name=node_name)
                ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, asset_name)
            assets_collection = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
        main_op = sess.graph.get_operation_by_name(init_op.name) if init_op else None
        if gfile.Exists(save_path):
            gfile.DeleteRecursively(save_path)
        builder = saved_model_builder.SavedModelBuilder(save_path)
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=new_mgd.meta_info_def.tags,
            signature_def_map=new_mgd.signature_def,
            assets_collection=assets_collection,
            main_op=main_op,
        )
        builder.save()
        if len(ev_opt_dict) > 0:
            target_path = os.path.join(save_path, 'variables')
            gfile.DeleteRecursively(target_path)
            _recursive_copy(os.path.dirname(variable_path), target_path)
            gfile.DeleteRecursively(os.path.dirname(variable_path))
        gfile.DeleteRecursively(tmp_path)

        print('Optmization Result:')
        for key, value in dense_opt_dict.items():
            print('Optimize dense op to {}: {}'.format(value[-1], key))
        for key, value in embed_opt_dict.items():
            print('Optimize embedding to {}: {}'.format(value[-1], key))
        for key, value in ev_opt_dict.items():
            print('Optimize embedding variable to {}: {}'.format(value[-1], key))


def convert_ckpt(ckpt_prefix, save_prefix, opt_model_path):
    opt_variables_path = os.path.join(opt_model_path, 'variables/variables')
    opt_variables = [name for name, _ in list_variables(opt_variables_path)]
    input_variables = [name for name, _ in list_variables(ckpt_prefix)]
    # Find embedding variables
    v2_quant_dict = {INT8: [], BF16: [], FP16: []}
    ev_quant_dict = {INT8: [], BF16: [], FP16: []}
    updated_variables = list(set(opt_variables).difference(set(input_variables)))
    for var_name in updated_variables:
        suffix = var_name.split('-')[-1]
        suffixes = [s for s in ev_suffix if s != 'values']
        if suffix in suffixes or var_name.endswith('/int8_scale'):
            continue
        is_qvar = False
        for opt_dtype in [INT8, BF16, FP16]:
            if var_name.endswith('/{}_data'.format(opt_dtype.lower())):
                v2_quant_dict[opt_dtype].append(var_name)
                is_qvar = True
            if var_name.endswith('/{}_data-values'.format(opt_dtype.lower())):
                ev_quant_dict[opt_dtype].append(var_name[:-7])
                is_qvar = True
        if not is_qvar:
            print('Not found variable: {}'.format(var_name))
    # Update checkpoint
    variable_path = ckpt_prefix
    for quant_dict, is_ev in [(v2_quant_dict, False), (ev_quant_dict, True)]:
        for opt_dtype, quant_names in quant_dict.items():
            if len(quant_names) == 0:
                continue
            len_suffix = len('/{}_data'.format(opt_dtype.lower()))
            names = [name[:-len_suffix] for name in quant_names]
            if opt_dtype == INT8:
                scale_names = ['{}/int8_scale'.format(name) for name in names]
            else:
                scale_names = []
            tmp_path = tempfile.mkdtemp(dir='.')
            opt_path = '{}/variables'.format(tmp_path)
            dtype = get_tf_dtype(opt_dtype)
            quantize_embedding_variable.quantize_by_name(
                variable_path, opt_path, names, quant_names, scale_names, dtype, is_ev
            )
            if variable_path != ckpt_prefix:
                gfile.DeleteRecursively(os.path.dirname(variable_path))
            variable_path = opt_path
    # Remove unused variables
    cur_variables = [name for name, _ in list_variables(variable_path)]
    removed_variables = list(set(cur_variables).difference(set(opt_variables)))
    save_dir = os.path.dirname(save_prefix)
    if not gfile.Exists(save_dir):
        gfile.MakeDirs(save_dir)
    quantize_embedding_variable.remove_variables_by_name(
        variable_path, save_prefix, removed_variables
    )
    if variable_path != ckpt_prefix:
        gfile.DeleteRecursively(os.path.dirname(variable_path))
    print('Convert Result:')
    for opt_dtype, quant_names in v2_quant_dict.items():
        for name in quant_names:
            print('Convert embedding to {}: {}'.format(opt_dtype, name))
    for opt_dtype, quant_names in ev_quant_dict.items():
        for name in quant_names:
            print('Convert embedding variable to {}: {}'.format(opt_dtype, name))


def _recursive_copy(src_dir, dest_dir):
    """Copy the contents of src_dir into the folder dest_dir.
    Args:
      src_dir: hdfs or local path.
      dest_dir: hdfs or local path.
    """
    for file_name in gfile.ListDirectory(src_dir):
        old_path = os.path.join(src_dir, file_name)
        new_path = os.path.join(dest_dir, file_name)

        if gfile.IsDirectory(old_path):
            _recursive_copy(old_path, new_path)
        elif not gfile.Exists(new_path):
            path_par = os.path.dirname(new_path)
            if not gfile.Exists(path_par):
                gfile.MakeDirs(path_par)
            print("copy {} to {}".format(old_path, new_path))
            gfile.Copy(old_path, new_path, overwrite=True)


"""
if __name__ == '__main__':
    import sys
    model_path = sys.argv[1]
    save_path = sys.argv[2]
    data_type = sys.argv[3]
    calib_file = sys.argv[4] if len(sys.argv) > 4 else None
    opt_dict = None
    optimize(model_path, save_path, opt_dict, data_type, calib_file)
"""
  parser.add_argument(
      '--model_path',
      type=str,
      required=True,
      help='directory containing the SavedModel to optimize')
  parser.add_argument(
      '--save_path',
      type=str,
      required=True,
      help='output directory for the optimized SavedModel')
  parser.add_argument(
      "--data_type",
      type=str,
      default=str(dtypes.float32.as_datatype_enum),
      help="""\
      The AttrValue enum to use for placeholders.
      Or a comma separated list, one value for each placeholder.\
      """)
  parser.add_argument(




def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input",
      type=str,
      default="",
      help="TensorFlow \'GraphDef\' file to load.")
  parser.add_argument(
      "--output",
      type=str,
      default="",
      help="File to save the output graph to.")
  parser.add_argument(
      "--input_names",
      type=str,
      default="",
      help="Input node names, comma separated.")
  parser.add_argument(
      "--output_names",
      type=str,
      default="",
      help="Output node names, comma separated.")
  parser.add_argument(
      "--frozen_graph",
      nargs="?",
      const=True,
      type="bool",
      default=True,
      help="""\
      If true, the input graph is a binary frozen GraphDef
      file; if false, it is a text GraphDef proto file.\
      """)
  parser.add_argument(
      "--placeholder_type_enum",
      type=str,
      default=str(dtypes.float32.as_datatype_enum),
      help="""\
      The AttrValue enum to use for placeholders.
      Or a comma separated list, one value for each placeholder.\
      """)
  parser.add_argument(
      "--toco_compatible",
      type=bool,
      default=False,
      help="""\
      If true, only use ops compatible with Tensorflow
      Lite Optimizing Converter.\
      """)
  return parser.parse_known_args()


if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
