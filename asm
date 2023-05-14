import json
import math
import os
import re
import sys

import torch
import random
import onnx
import logging
from math import ceil
import numpy as np
import pandas as pd

# file_path = os.path.abspath(__file__)
# rootpath = os.sep.join(file_path.split(os.sep)[:file_path.split(os.sep).index('workspace') + 2]) + os.sep
# sys.path.append(f'{rootpath}XpengRT/python/xperngrt/lib')
# sys.path.append(f'{rootpath}XpengRT/python/xpengrt/scripts/')
# sys.path.append(f'{rootpath}XpengRT/xpengrt/backend/lib/Analysis/assembler/build/python/')

sys.path.append("/home/jenkins/bin/XpengRT/python/xpengrt/lib")
sys.path.append("/home/jenkins/bin/XpengRT/python/xpengrt/scripts/")
sys.path.append("/home/jenkins/bin/XpengRT/xpengrt/backend/lib/Analysis/assembler/build/python/")
from data_package import *
from pipeline import *
import parse_pb

L3_BUFFER = 'L3'
L2_BUFFER = 'L2'
L1_BUFFER = 'L1'
L1_BUFFER_SIZE = 6 * (1 << 20)  # 6MB
ELEMENT_WISE = ['add', 'mul', 'max', 'min']
FUSED = ['relu', 'sigmoid', 'tanh']
OP_MAP = {'add': 'vsadd', 'min': 'vmul', 'max': 'vmax'}
ALIGN_TYPE_MAP = {'ifm': 'c16w32', 'weight': 'kernel', 'bias': 'bias', 'quant': 'quant', 'dequant': 'dequant',
                  'ofm': 'c32w32', 'elt': 'c32w32'}

IFM_SHAPE = {'i8': 'c16w32', 'i10': 'c8w32', 'i16': 'c8w32'}
DATA_TYPE = {'i8': "int8", 'i10': "int10", 'i16': "int16", 'i32': "int32"}
DATA_TYPES = {'i8': np.int8, 'i10': np.int16, 'i16': np.int16, 'i32': np.int32}

VERSION_INFO = ".version 0.1.2\n\n"
env_config = {'op_asm_path': './op_asm_test',
              'veu_op_asm_path': './veu_op_asm',
              'cmodel_path': './cmodel',
              'standalone_mask_fill_path': './standalone_mask_fill',
              'tool_bin_path': './xpengrt-as'
              }
# for 'h_size', 'w_size/m_size', 'ic_size/k_size', 'oc_size'
size_field_list = ['h_size', 'w_size/m_size', 'ic_size/k_size', 'oc_size']

# limit field
field_list = {
    'operator': ['Conv2d', 'Conv', 'winoconv', 'matmul', 'maxpool', 'avgpool', 'globalavgpool', 'depthwise', 'resize',
                 'mul', 'fusedConvAdd', 'fusedConvMax', 'fusedConvMul', 'fusedConvMin', 'FC', 'fusedMulAdd',
                 'gridSample', 'pixshuffer'],
    'attributes': '',
    'fused': ['', 'relu', 'sigmoid', 'tanh', 'add', 'max', 'min', 'mul', 'relu+add', 'relu+max', 'relu+min', 'relu+mul',
              'add+relu', 'max+relu', 'min+relu', 'mul+relu', 'add+sigmoid', 'max+sigmoid', 'min+sigmoid',
              'mul+sigmoid', 'sigmoid+mul', 'add+tanh', 'max+tanh', 'min+tanh', 'mul+tanh', 'tanh+mul',
              'relu+add+sigmoid', 'relu+max+sigmoid', 'relu+min+sigmoid', 'relu+mul+sigmoid', 'relu+add+tanh',
              'relu+max+tanh', 'relu+min+tanh', 'relu+mul+tanh', 'add+relu+sigmoid', 'max+relu+sigmoid',
              'min+relu+sigmoid', 'mul+relu+sigmoid', 'add+relu+tanh', 'max+relu+tanh', 'min+relu+tanh',
              'mul+relu+tanh'],
    'batch': [1, ],
    'h_size': [1, 8192],
    'w_size/m_size': [1, 8192],
    'ic_size/k_size': [1, 8192],
    'oc_size': [1, 8192],
    'bias': [1, 0],
    'dqt': ['tmu', 'elt', 'both'],
    'scale': ['', 1, ],
    'dqt_prec': ['', 1, ],
    'ofm_prec': ['', 1, ],
    'elt_prec': ['', 1, ],
    'ifm_prec': ['int8', 'int10', 'int8&int10', 'multiple'],
    'kernel': [1, 2, 3, 4, 5, 7],  # TODO: need to confirm
    'stride': [1, 2],
    'padding': [0, 1, 2, 3],
    'precision': ['i8', 'i10', 'multiple'],
    'misc_en': [1, 0]
}

# Bit
align_config = {
    # 1.ifm_size: nchw --> c16w32
    'ifm': {
        'ic': 16,
        'iw': 32
    },
    # 2.ofm_size: c32w32
    'ofm': {
        'oc': 32,
        'ow': 32
    },
    # 3.weight_size
    # 3.1 kernel_size  [oc, ic, kh, kw] -> int8 [oc/32, ic/16, kh, kw, 32, 16]
    # 3.2 bias_size  int32 -> 4B
    # 3.3 quant_size  int32 -> 4B
    'weights': {
        'kernel': {
            'oc': 32,
            'ic': 16,
        },
        'bias': 32,
        'quant': 32,
        'align': 4096
    }
}

precion = {
    'ifm': ['i8', 'i10'],
    'ofm': ['i8', 'i10', 'i16'],
}

# for veu op file
VEU_OP_DTYPE = ["i10+i8", "i8+i10", "i8+i8", "i10+i10", "i16+i16"]
SPECIAL_OP = ["mul", "mul+relu", "mul+sigmoid_lut", "mul+tanh_lut", "mul+reciprocal", "mul+sqrt", "sigmoid_lut+mul",
              "tanh_lut_mul", "reciprocal+mul", "tanh_lut_mul", "sqrt+mul", "swish"]
FUSED_VEU_OP = {
    "add": {"data_type": "", "type_ins": "sadd", "func": "fused_elt_same_w"},
    "sub": {"data_type": "", "type_ins": "ssub", "func": "fused_elt_same_w"},

    "mul": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "", "func": "fused_elt_mul_mix_v31i16"},
            {"data_type": VEU_OP_DTYPE[1], "type_ins": "", "func": "fused_elt_mul_mix_v30i16"},
            {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_elt_same_w"},
            {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_elt_same_w"},
            {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_elt_same_w"}],
    "max": {"data_type": "", "type_ins": "max", "func": "fused_elt_same_w"},
    "min": {"data_type": "", "type_ins": "min", "func": "fused_elt_same_w"},
    "relu": {"data_type": "", "type_ins": "", "func": "fused_relu"},
    "sigmoid_lut": {"data_type": "", "type_ins": "", "func": "fused_lut"},
    "tanh_lut": {"data_type": "", "type_ins": "", "func": "fused_lut"},
    "reciprocal": {"data_type": "", "type_ins": "", "func": "fused_lut"},
    "sqrt": {"data_type": "", "type_ins": "", "func": "fused_lut"},
    "avg_pooling": {"data_type": "", "type_ins": "", "func": "fused_avg_pooling"},
    "max_pooling": {"data_type": "", "type_ins": "", "func": "fused_max_pooling"},
    "global_avg_pooling": {"data_type": "", "type_ins": "", "func": "fused_avg_pooling"},
    "add+relu": {"data_type": "", "type_ins": "sadd", "func": "fused_elt_relu_same_w"},
    "min+relu": {"data_type": "", "type_ins": "min", "func": "fused_elt_relu_same_w"},
    "max+relu": {"data_type": "", "type_ins": "max", "func": "fused_elt_relu_same_w"},

    "mul+relu": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "", "func": "fused_mul_relu_mix_v31i16"},
                 {"data_type": VEU_OP_DTYPE[1], "type_ins": "", "func": "fused_mul_relu_mix_v30i16"},
                 {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_elt_relu_same_w"},
                 {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_elt_relu_same_w"},
                 {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_elt_relu_same_w"}],
    "relu+add": {"data_type": "", "type_ins": "sadd", "func": "fused_relu_add_max_min"},
    "relu+max": {"data_type": "", "type_ins": "max", "func": "fused_relu_add_max_min"},
    "relu+min": {"data_type": "", "type_ins": "min", "func": "fused_relu_add_max_min"},
    "add+sigmoid_lut": {"data_type": "", "type_ins": "sadd", "func": "fused_elt_lut_same_w"},
    "add+tanh_lut": {"data_type": "", "type_ins": "sadd", "func": "fused_elt_lut_same_w"},
    "min+sigmoid_lut": {"data_type": "", "type_ins": "min", "func": "fused_elt_lut_same_w"},
    "min+tanh_lut": {"data_type": "", "type_ins": "min", "func": "fused_elt_lut_same_w"},
    "max+sigmoid_lut": {"data_type": "", "type_ins": "max", "func": "fused_elt_lut_same_w"},
    "max+tanh_lut": {"data_type": "", "type_ins": "max", "func": "fused_elt_lut_same_w"},
    "add+reciprocal": {"data_type": "", "type_ins": "sadd", "func": "fused_elt_lut_same_w"},
    "add+sqrt": {"data_type": "", "type_ins": "sadd", "func": "fused_elt_lut_same_w"},

    "mul+sigmoid_lut": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "", "func": "fused_mul_lut_mix_v31i16"},
                        {"data_type": VEU_OP_DTYPE[1], "type_ins": "", "func": "fused_mul_lut_mix_v30i16"},
                        {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_elt_lut_same_w"},
                        {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_elt_lut_same_w"},
                        {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_elt_lut_same_w"}],
    "mul+tanh_lut": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "", "func": "fused_mul_lut_mix_v31i16"},
                     {"data_type": VEU_OP_DTYPE[1], "type_ins": "", "func": "fused_mul_lut_mix_v30i16"},
                     {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_elt_lut_same_w"},
                     {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_elt_lut_same_w"},
                     {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_elt_lut_same_w"}],
    "mul+reciprocal": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "", "func": "fused_mul_lut_mix_v31i16"},
                       {"data_type": VEU_OP_DTYPE[1], "type_ins": "", "func": "fused_mul_lut_mix_v30i16"},
                       {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_elt_lut_same_w"},
                       {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_elt_lut_same_w"},
                       {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_elt_lut_same_w"}],
    "mul+sqrt": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "", "func": "fused_mul_lut_mix_v31i16"},
                 {"data_type": VEU_OP_DTYPE[1], "type_ins": "", "func": "fused_mul_lut_mix_v30i16"},
                 {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_elt_lut_same_w"},
                 {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_elt_lut_same_w"},
                 {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_elt_lut_same_w"}],
    "sigmoid_lut+mul": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "mul16.vf2", "func": "fused_lut_mul_v30i16"},
                        {"data_type": VEU_OP_DTYPE[1], "type_ins": "mul16.vf2", "func": "fused_lut_mul_v30i8"},
                        {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_lut_mul_v30i16"},
                        {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_lut_mul_v30i16"},
                        {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_lut_mul_v30i16"}],
    "tanh_lut_mul": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "mul16.vf2", "func": "fused_lut_mul_v30i16"},
                     {"data_type": VEU_OP_DTYPE[1], "type_ins": "mul16.vf2", "func": "fused_lut_mul_v30i8"},
                     {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_lut_mul_v30i16"},
                     {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_lut_mul_v30i16"},
                     {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_lut_mul_v30i16"}],
    "reciprocal+mul": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "mul16.vf2", "func": "fused_lut_mul_v30i16"},
                       {"data_type": VEU_OP_DTYPE[1], "type_ins": "mul16.vf2", "func": "fused_lut_mul_v30i8"},
                       {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_lut_mul_v30i16"},
                       {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_lut_mul_v30i16"},
                       {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_lut_mul_v30i16"}],
    "sqrt+mul": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "mul16.vf2", "func": "fused_lut_mul_v30i16"},
                 {"data_type": VEU_OP_DTYPE[1], "type_ins": "mul16.vf2", "func": "fused_lut_mul_v30i8"},
                 {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_lut_mul_v30i16"},
                 {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_lut_mul_v30i16"},
                 {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_lut_mul_v30i16"}],
    "swish": [{"data_type": VEU_OP_DTYPE[0], "type_ins": "", "func": "fused_swish_mul_mix_v31i16"},
              {"data_type": VEU_OP_DTYPE[1], "type_ins": "", "func": "fused_swish_mul_mix_v31i8"},
              {"data_type": VEU_OP_DTYPE[2], "type_ins": "mul8", "func": "fused_swish_mul_same_w"},
              {"data_type": VEU_OP_DTYPE[3], "type_ins": "mul16", "func": "fused_swish_mul_same_w"},
              {"data_type": VEU_OP_DTYPE[4], "type_ins": "mul16", "func": "fused_swish_mul_same_w"}],

    "h_swish": {"data_type": "", "type_ins": "", "func": "fused_h_swish"},
    "relu6": {"data_type": "", "type_ins": "", "func": "fused_relu6"},
    "div": {"data_type": "", "type_ins": "", "func": "fused_relu6"},
    "layernorm_reciprocal": {"data_type": "", "type_ins": "", "func": "fused_relu6"},
    "matrix_mul_vector": {"data_type": "", "type_ins": "", "func": "standalone_matrixmulscalar"}
}


def logging_func():
    print('coding line:{}, function name: {}'.format(sys._getframe().f_lineno, sys._getframe().f_code.co_name))
    print('parent func:{}'.format(sys._getframe(1).f_code.co_name))


def case_format_check(idx, rowdata):
    error_flag = False
    debug_info = False
    error_key = ''
    error_value = ''

    for key, value in rowdata:
        if key.lower() == 'attributes':
            new_value = value.strip()
            result = re.match(r'\d[xX]\d (Deconvolution|Convolution), Stride\(\d,\d\), Padding\(\d,\d\)$', new_value,
                              flags=re.I)
            if not result:
                debug_info = True
                error_key = key
                error_value = value
                error_flag = True
                break
            continue
        if key.lower() in size_field_list:
            size_separator = ':'
            if size_separator in value:
                size_min = int(value.split(size_separator)[0])
                size_max = int(value.split(size_separator)[1])
                if not (size_min >= field_list[key.lower()][0] and size_max <= field_list[key.lower()][1]):
                    debug_info = True
                    error_key = key
                    error_value = value
                    error_flag = True
                    break
            else:
                if not (value > field_list[key.lower()] and value < field_list[key.lower()]):
                    debug_info = True
                    error_key = key
                    error_value = value
                    error_flag = True
                    break
            continue

        # if key.lower() == 'dqt':
        #     # TODO
        #     debug_info = True
        #     error_key = key
        #     error_value = value
        #     continue
        #
        # if key.lower() in ['scale', 'dqt_prec', 'ofm_prec', 'elt_prec', 'ifm_prec']:
        #     # TODO  Not in field
        #     continue

        if 'padding' in key.lower():
            new_value = value.strip()
            result = re.match(r'\[\d, \d, \d, \d\]$', new_value, flags=re.I)
            if not result:
                debug_info = True
                error_key = key
                error_value = value
                error_flag = True
                break
            continue
        if str(value).lower() not in [str(ele).lower() for ele in field_list[key.lower()]]:
            debug_info = True
            error_key = key
            error_value = value
            error_flag = True
            break

    if debug_info:
        print("Please check row %s " % idx)
        print("format of %s mismatch: %s" % (error_key, error_value))
    return error_flag


def cal_conv_output_size(ih, iw, k, p, s):  # todo: Detailed by situation
    oh = (ih + 2 * p - k) // s + 1
    ow = (iw + 2 * p - k) // s + 1
    return oh, ow


def _align_shape(shape, align):
    return ceil(shape / align) * align


def check_memory(bt, k, ih, iw, ic, oh, ow, oc, ifm_pre, ofm_pre):  # todo: ih, iw, oh, ow
    # 1.ifm_size: nchw --> c16w32
    ifm_ic = align_config['ifm']['ic']
    ifm_iw = align_config['ifm']['iw']
    # 2.ofm_size: c32w32
    ofm_oc = align_config['ofm']['oc']
    ofm_ow = align_config['ofm']['ow']
    weights_kernel_oc = align_config['weights']['kernel']['oc']
    weights_kernel_ic = align_config['weights']['kernel']['ic']
    weights_bias = align_config['weights']['bias']
    weights_quant = align_config['weights']['quant']
    weights_align = align_config['weights']['align']
    ifm_align_byte = 1 if (ifm_pre.strip() == 'i8') else 2  # TODO: only support int8/int10
    ofm_align_byte = 1 if (ofm_pre.strip() == 'i8') else 2

    # bit align
    ifm_align_size = bt * _align_shape(ic, ifm_ic) * ih * _align_shape(iw, ifm_iw) * ifm_align_byte

    ofm_align_size = bt * _align_shape(oc, ofm_oc) * oh * _align_shape(ow, ofm_ow) * ofm_align_byte

    # 3.weight_size
    # 3.1 kernel_size  [oc, ic, kh, kw] -> int8 [oc/32, ic/16, kh, kw, 32, 16] // todo:int16
    # default int8 -> 1B
    kernel_align_size = _align_shape(ic, weights_kernel_ic) * _align_shape(oc,
                                                                           weights_kernel_oc) * k * k * ifm_align_byte

    # 3.2 bias_size   8bit -> 1Byte
    bias_align_size = _align_shape(oc, weights_bias) * int(weights_bias / 8)

    # 3.3 quant_size
    quant_align_size = _align_shape(oc, weights_quant) * int(weights_bias / 8)

    weight_size = (kernel_align_size + bias_align_size + quant_align_size)  # 4KB
    if (ifm_align_size + ofm_align_size + weight_size) > L1_BUFFER_SIZE:
        return False

    return True


def random_size(rd_content, random_case=0):
    each_col_size_vector = list()
    batch = rd_content['batch']
    kernel = rd_content["kernel"]
    padding = [int(i.strip()) for i in rd_content["padding [pad_t, pad_d, pad_l, pad_r]"].strip()[1:-1].split(',')][0]
    stride = rd_content["stride"]
    ifm_prec = rd_content["ifm_prec"]
    ofm_prec = rd_content["ofm_prec"]
    h_size = rd_content["h_size"].split(':')
    w_size = rd_content["w_size/m_size"].split(':')
    ic_size = rd_content["ic_size/k_size"].split(':')
    oc_size = rd_content["oc_size"].split(':')

    ih = random.randint(int(h_size[0]), int(h_size[1]))
    iw = random.randint(int(w_size[0]), int(w_size[1]))
    ic = random.randint(int(ic_size[0]), int(ic_size[1]))
    oc = random.randint(int(oc_size[0]), int(oc_size[1]))
    # 随机值
    while len(each_col_size_vector) < random_case:
        oh, ow = cal_conv_output_size(ih, iw, kernel, padding, stride)
        if check_memory(batch, kernel, ih, iw, ic, oh, ow, oc, ifm_prec, ofm_prec):
            each_col_size_vector.append([batch, kernel, ih, iw, ic, oh, ow, oc])
        else:
            ih = random.randint(int(h_size[0]), ih)
            iw = random.randint(int(w_size[0]), iw)
            ic = random.randint(int(ic_size[0]), ic)
            oc = random.randint(int(oc_size[0]), oc)

    # 边界值
    for ih in [int(h_size[0]), int(h_size[1])]:
        for iw in [int(w_size[0]), int(w_size[1])]:
            for ic in [int(ic_size[0]), int(ic_size[1])]:
                for oc in [int(oc_size[0]), int(oc_size[1])]:
                    oh, ow = cal_conv_output_size(ih, iw, kernel, padding, stride)
                    if check_memory(batch, kernel, ih, iw, ic, oh, ow, oc, ifm_prec, ofm_prec):
                        each_col_size_vector.append([batch, kernel, ih, iw, ic, oh, ow, oc])

    return each_col_size_vector


def random_input(shape, dtype=torch.float32):
    x = ''
    if dtype == "i8" or dtype == 'int8':
        x = np.random.randint(-128, 127, shape, dtype=np.int8)
    elif dtype == "i10" or dtype == 'int10':
        x = np.random.randint(-512, 511, shape, dtype=np.int16)
    elif dtype == "i16" or dtype == 'int16':
        x = np.random.randint(-32768, 32767, shape, dtype=np.int16)
    elif dtype == "i32" or dtype == 'int32':
        x = np.random.randint(-2147483648, 2147483647, shape, dtype=np.int32)
    elif dtype == "u32":
        x = np.random.randint(1, 10, shape, dtype=np.uint32)
    else:
        x = (torch.randn(shape, dtype=dtype).detach().numpy() - 0.5) * 4
    return x


def env_init(op_file):
    for key, value in env_config.items():
        if not os.path.exists(value):
            print(value)
            if key == 'op_asm_path':
                os.makedirs(value, exist_ok=True)
                continue
            print('Please check %s path,it is not exist' % key)
            return False

    if not os.path.exists(op_file):
        print('%s is not exist' % op_file)
        return False

    return True


class Buffer(object):

    def __init__(self, buf_type, buf_name, format, shape, dtype, addr, align_size):
        if buf_type == "L3":
            self.buf_type = ".param.global "
            if buf_name == "ofm":
                self.buf_name = "output"
            else:
                self.buf_name = buf_name + "_l3"
        elif buf_type == "L2":
            self.buf_type = ".param.shared "
            self.buf_name = buf_name + "_l2"
        else:
            self.buf_type = ".param.local "
            self.buf_name = buf_name + "_l1"
        self.format = format
        self.shape = shape
        self.dtype = dtype
        self.addr = addr
        self.align_size = align_size

    def gen_buffer(self):
        stmt = ''
        stmt += self.buf_type + ''
        stmt += self.buf_name + "("
        stmt += self.format + ', '
        stmt += self.shape + ', '
        stmt += self.dtype + ', '
        stmt += self.addr + ', '
        stmt += self.align_size + ");"
        return stmt


class Scalar(object):
    def __init__(self, s_name, s_value):
        self.base = '.param.scalar'
        self.s_name = s_name
        self.s_value = s_value

    def gen_scalar(self):
        stmt = ''
        stmt += self.base + ' '
        stmt += str(self.s_name) + '('
        if isinstance(self.s_value, str):
            stmt += '"' + self.s_value + '")'
        elif isinstance(self.s_value, int):
            stmt += str(self.s_value) + ')'
        stmt += ';'
        return stmt


class Config(object):
    def __init__(self, npu_num, paramAddr, ofmAddr):
        self.base = ".config "
        self.npu_num = npu_num
        self.paramAddr = paramAddr
        self.ofmAddr = ofmAddr
        self.config_vec = []

    def gen_config(self):
        self.config_vec.append(self.base + 'npu(' + str(self.npu_num) + ");")
        self.config_vec.append(self.base + 'paramAddr(' + self.paramAddr + ");")
        self.config_vec.append(self.base + 'ofmAddr(' + self.ofmAddr + ");")
        return self.config_vec


class Entry(object):
    def __init__(self, cdtu_stream, npu_stream, veu_op=None):
        self.base = ".entry "
        self.cdtu_stream = cdtu_stream
        self.npu_stream = npu_stream
        self.veu_op = veu_op
        self.entry_vec = []

    def gen_entry(self):
        self.entry_vec.append(self.base + 'cdtu_stream(' + self.cdtu_stream + ");")
        self.entry_vec.append(self.base + 'npu_stream(' + self.npu_stream + ");")
        logging_func()
        print(self.veu_op)
        if self.veu_op:
            self.entry_vec.append(self.base + self.veu_op + '(veu);')
        else:
            self.entry_vec.append(self.base + "fused_avg_pooling(veu);")
        return self.entry_vec


class Sync(object):
    def __init__(self, idx, src, dst):
        self.idx = idx
        self.src = src
        self.dst = dst

    def gen_sync(self):
        stmt = ".param.sync message" + str(self.idx) + '(' + str(self.idx) + ','
        stmt += "src(" + self.src + '),'
        stmt += 'dst(' + self.dst + '));'
        return stmt


class TensorInfo(object):
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class GenConvParamsData(object):
    def __init__(self, meta_path, ws, onnx_name):
        self.optype = ''
        self.input_pramas = []
        self.tensor_dict = {}
        self.dilations = [1, 1]
        self.group = 1
        self.padding = ''
        self.strides = []
        self.ws = ws
        self.meta_path = meta_path
        self.onnx_name = self.ws + '/' + onnx_name
        self.load_json(meta_path)

    def load_json(self, meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            self.optype = metadata['op']
            self.input_pramas = metadata['input_params']
            self.padding = metadata['padding']
            self.strides = metadata['stride']
            for item in self.input_pramas:
                self.tensor_dict[item] = TensorInfo(metadata[item]["name"], metadata[item]["shape"],
                                                    metadata[item]["dtype"])

    def parse_input(self):
        inputs = []
        outputs = ["output"]
        for item in self.input_pramas:
            if item in ["quant", "dequant", 'ofm']:
                continue
            if item == "ifm":
                inputs.append("input")
            else:
                inputs.append(item)
        return inputs, outputs

    def pack_tensor(self, values):
        w_align = 32
        w_slice = 32
        if values.dtype.name == 'int8':
            c_align = 16
            c_slice = 4
        elif values.dtype.name == 'int16':
            c_align = 8
            c_slice = 2
        else:
            raise TypeError(f"weight data type error, expect int8/int16, got {values.dtype.name} instead!")
        # [N, C, H, W]
        pad_w = _align_shape(values.shape[3], w_align) - values.shape[3]
        pad_c = _align_shape(values.shape[1], c_align) - values.shape[1]
        if pad_w or pad_c:
            values = np.pad(values, ((0, 0), (0, pad_c), (0, 0), (0, pad_w)), constant_values=(0, 0))

        # int8 [N, H, W//32, C//4, C4, W32]
        values = values.transpose(0, 2, 3, 1)
        values = values.reshape(
            values.shape[0], values.shape[1],
            values.shape[2] // w_slice, w_slice,
            values.shape[3] // c_slice, c_slice
        ).transpose(0, 1, 2, 4, 5, 3)
        return values.tobytes()

    def run(self):
        inputs = [
            onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, shape=self.tensor_dict['ifm'].shape)]
        outputs = [
            onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, shape=self.tensor_dict['ofm'].shape)]
        inputs_vec, outputs_vec = self.parse_input()
        node_1 = onnx.helper.make_node(
            "Conv",
            ['input', 'weight0', 'bias0'],
            ['conv0_out'],
            "Conv_0",
            dilations=[1, 1],
            group=1,
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1]
        )
        node_2 = onnx.helper.make_node(
            "Conv",
            ['conv0_out', 'weight1', 'bias1'],
            ['output'],
            "Conv_1",
            dilations=[1, 1],
            group=1,
            kernel_shape=[self.tensor_dict['weight'].shape[2], self.tensor_dict['weight'].shape[3]],
            pads=self.padding,
            strides=self.strides
        )

        input_data = np.random.rand(self.tensor_dict['ifm'].shape[0],
                                    self.tensor_dict['ifm'].shape[1], self.tensor_dict['ifm'].shape[2],
                                    self.tensor_dict['ifm'].shape[3]).astype(np.float32)

        kernel_shape = self.tensor_dict['weight'].shape
        k1_shape = [kernel_shape[1], kernel_shape[1], 1, 1]
        k2_shape = kernel_shape
        initialize = [
            onnx.helper.make_tensor(
                'weight0', onnx.TensorProto.FLOAT, k1_shape,
                np.random.rand(*k1_shape).astype(np.float32).flatten().tolist()),
            onnx.helper.make_tensor(
                'bias0', onnx.TensorProto.FLOAT, [k1_shape[0]],
                np.random.rand(k1_shape[0]).astype(np.float32).flatten().tolist()),
            onnx.helper.make_tensor(
                'weight1', onnx.TensorProto.FLOAT, k2_shape,
                np.random.rand(*k2_shape).astype(np.float32).flatten().tolist()),
            onnx.helper.make_tensor(
                'bias1', onnx.TensorProto.FLOAT, self.tensor_dict['bias'].shape,
                np.random.rand(self.tensor_dict['bias'].shape[0]).astype(np.float32).flatten().tolist())]
        graph = onnx.helper.make_graph([node_1, node_2],
                                       self.onnx_name,
                                       inputs,
                                       outputs,
                                       initializer=initialize)

        bcast_model = onnx.helper.make_model(graph)
        onnx.save(bcast_model, self.onnx_name + ".onnx")

        dataset = {}
        import xpengrt
        dataset["input"] = xpengrt.Tensor(input_data)

        graph = xpengrt.Graph()
        xpengrt.compile(graph, self.onnx_name + ".onnx")
        table_name = self.onnx_name + ".table"
        data_loader = []
        data_loader.append(dataset)
        xpengrt.quantize(graph, data_loader, table_name)
        if self.tensor_dict['ifm'].dtype in ["i10"] or self.tensor_dict['ofm'].dtype in ["i10", 'i16']:
            with open(table_name, 'r') as f:
                table = [item.strip() for item in f.read().strip().split('\n') if item.strip()]
            if self.tensor_dict['ifm'].dtype in ["i10"]:
                table[1] += " " + DATA_TYPE[self.tensor_dict['ifm'].dtype]
            if self.tensor_dict['ofm'].dtype in ["i10", 'i16']:
                table[2] += " " + DATA_TYPE[self.tensor_dict['ofm'].dtype]
            with open(table_name, 'w') as f:
                f.write('\n'.join(table))
        graph = xpengrt.Graph()
        xpengrt.compile(graph, self.onnx_name + ".onnx")
        xpengrt.quantize_graph(graph, table_name, "")

        ir = xpengrt.IR(graph)
        ir.ir_write(self.onnx_name)
        from visualize_ir import json_to_onnx
        json_to_onnx(self.onnx_name + ".json", self.onnx_name + ".param")
        data = parse_pb.pb_to_np_dict(self.onnx_name + ".param")

        with open(self.ws + "/quant_0.bin", 'wb') as f:
            f.write(data["Conv_1_quant_params"].tobytes())

        with open(self.ws + "/weight_0.bin", 'wb') as f:
            f.write(data["weight1"].tobytes())

        with open(self.ws + "/bias_0.bin", 'wb') as f:
            f.write(data["bias1"].tobytes())

        ifm_tensor = ''
        with open(self.ws + "/ifm_0.bin", 'wb') as f:
            ifm_tensor = random_input(self.tensor_dict['ifm'].shape, self.tensor_dict['ifm'].dtype)
            f.write(ifm_tensor.tobytes())
        import goldengen
        ofm_goldengen = goldengen.Gengolden()
        ofm_goldengen.run(self.ws, self.meta_path)
        print("ofm goldengen  succcess----------")

        # ifm (nchw ->  c16w32/c8w32)
        with open(self.ws + '/quant_tensor_input_c16w32_cmodel.bin', 'wb') as f:
            f.write(self.pack_tensor(ifm_tensor))

        # ofm (nchw -> c32w32)
        ofm_tensor = np.fromfile(self.ws + "/ofm_0_c32w32_" + self.tensor_dict['ofm'].dtype + ".bin",
                                 dtype=DATA_TYPES[self.tensor_dict['ofm'].dtype]).reshape(self.tensor_dict['ofm'].shape)
        pad_c = _align_shape(self.tensor_dict['ofm'].shape[1], 32) - self.tensor_dict['ofm'].shape[1]
        pad_w = _align_shape(self.tensor_dict['ofm'].shape[3], 32) - self.tensor_dict['ofm'].shape[3]
        if pad_c or pad_w:
            ofm_tensor = np.pad(ofm_tensor, ((0, 0), (0, pad_c), (0, 0), (0, pad_w)), constant_values=(0, 0))

        ofm_tensor = ofm_tensor.transpose(0, 2, 3, 1)
        ofm_tensor = ofm_tensor.reshape(ofm_tensor.shape[0], ofm_tensor.shape[1], ofm_tensor.shape[2] // 32, 32,
                                        ofm_tensor.shape[3] // 4, 4).transpose(0, 1, 2, 4, 5, 3)
        with open(self.ws + '/dequant_tensor_output_c32w32_cmodel.bin', 'wb') as f:
            f.write(ofm_tensor.tobytes())
        # weight
        weight_align_cache = DataPackage().pack_kernel(data["weight1"])
        bias_align_cache = DataPackage().pack_bias(data["bias1"])
        quant_align_cache = DataPackage().pack_quant(data["Conv_1_quant_params"])

        with open(os.path.join(self.ws, f"kernel_align.bin"), "wb") as f:
            f.write(weight_align_cache)
        with open(os.path.join(self.ws, f"bias_align.bin"), "wb") as f:
            f.write(bias_align_cache)
        with open(os.path.join(self.ws, f"quant_align.bin"), "wb") as f:
            f.write(quant_align_cache)

        weight_cache = b''
        weight_cache += weight_align_cache
        weight_cache += bias_align_cache
        weight_cache += quant_align_cache

        with open(os.path.join(self.ws, f"weight.bin"), 'wb') as f:
            f.write(weight_cache)
        with open(os.path.join(self.ws, f"read_file_list.txt"), 'w') as f:
            f.write("ifm_l3_input\n")
            f.write("quant_tensor_input_c16w32_cmodel.bin\n")
            f.write("tmu0_npu0_output\n")
            f.write("dequant_tensor_output_c32w32_cmodel.bin\n")
            f.write("ofm_l3_output\n")
            f.write("dequant_tensor_output_c32w32_cmodel.bin\n")


class ConvTestcase(object):
    def __init__(self, df_content):
        self.content = df_content
        self.operator = ''
        self.attributes = ''
        self.fused = ''
        self.batch = 1
        self.ih_size = 0
        self.iw_size = 0
        self.oh_size = 0
        self.ow_size = 0
        self.ic_size = 0
        self.oc_size = 0
        self.kernel = 1
        self.stride = 1
        self.padding = [0, 0, 0, 0]
        self.bias = 1
        self.elt = 0
        self.dqt = "both"
        self.winograd = False
        self.ifm_prec = ''
        self.scale = ''
        self.dqt_prec = ''
        self.ofm_prec = ''
        self.elt_prec = ''
        self.misc_en = 0

        self.paramAddr = ''
        self.ofmAddr = ''
        self.base_dir = ''
        self.ofm_align_shape = []
        self.fused_op = ''
        self.veu_op = ''
        self.loop_times = 0

        self.buf_sequence = []
        self.config_vec = []
        self.entry_vec = []
        self.scalar_vec = []
        self.buf_vec = []
        self.sync_vec = []
        self.cdtu_stream = []
        self.npu_stream = []
        self.fused_vec = []
        self.param = []
        self.prec_map = {"weight": 'i8', 'bias': 'i32', 'quant': 'i32', 'dequant': 'i32'}

    def handle_attrs(self, ifm_prec, ofm_prec, item_size):
        self.operator = "conv2d" if "winograd" in self.content["Operator"] else self.content["Operator"]
        self.winograd = True if "winograd" in self.content["Operator"] else False
        self.attributes = self.content["attributes"]
        self.batch = self.content["batch"]
        self.stride = self.content["stride"]
        self.bias = self.content["bias"]
        self.padding = [int(i.strip()) for i in
                        self.content["padding [pad_t, pad_d, pad_l, pad_r]"].strip()[1:-1].split(',')]
        # [n, k, ih, iw, ic, oh, ow, oc]
        self.fused = self.content["Fused"].split('+')
        self.raw_fused = self.content["Fused"]
        self.veu_op = ('fused_' + "_".join(self.fused)) if len(self.fused) else ''
        # self.fused = ['']  # TODO Fused
        if any(item in self.fused for item in ELEMENT_WISE):
            self.elt = 1
            self.prec_map['elt'] = ofm_prec
        self.dqt = self.content["dqt"]
        self.ifm_prec = ifm_prec
        self.ofm_prec = ofm_prec
        self.prec_map['ifm'] = ifm_prec
        self.prec_map['weight'] = ifm_prec
        self.prec_map['ofm'] = ofm_prec
        self.scale = self.content["scale"]
        self.dqt_prec = self.content["dqt_prec"]
        self.misc_en = self.content["misc_en"]

        self.batch, self.kernel, self.ih_size, self.iw_size, self.ic_size, \
            self.oh_size, self.ow_size, self.oc_size = item_size

        self.loop_times = (self.batch * self.oh_size * self.ow_size * self.oc_size + 127) // 128  # TODO 127/128 ???

    def create_op_dir(self):
        name = self.operator + "_"
        atts = self.attributes.split(',')
        for item in atts:
            if "Convolution" in item or "Deconvolution" in item:
                name += 'k' + str(self.kernel) + '_'
            if "Stride" in item:
                name += 's' + item.split('(')[1] + '_'  # TODO why is not self.stride?
            if "Padding" in item:
                name += "p" + str(item.split('(')[1])  # TODO why is not self.padding?

        # 不加asm的name
        name += '_n' + str(self.batch) + '_ic' + str(self.ic_size) + '_h' + str(self.ih_size) + '_w' + str(
            self.iw_size) + '_oc' + str(self.oc_size) + '_' + self.ifm_prec + '_' + self.ofm_prec
        self.base_dir = env_config['op_asm_path'] + "/" + name
        os.makedirs(self.base_dir, exist_ok=True)
        return name

    def get_dtype_size(self, dtype):
        dtype_size = 1
        if dtype == 'i8':
            dtype_size = 1
        elif dtype == 'i10' or dtype == 'i16':
            dtype_size = 2
        elif dtype == 'i32':
            dtype_size = 4
        else:
            logging.error(f'data type: {dtype} is not supported')
        return dtype_size

    # name, shape, prec, format, align_shape
    def cal_align_size(self, type_):
        ifm_ic = align_config['ifm']['ic']
        ifm_iw = align_config['ifm']['iw']
        # 2.ofm_size: c32w32
        ofm_oc = align_config['ofm']['oc']
        ofm_ow = align_config['ofm']['ow']
        weights_kernel_oc = align_config['weights']['kernel']['oc']
        weights_kernel_ic = align_config['weights']['kernel']['ic']
        weights_bias = align_config['weights']['bias']
        weights_quant = align_config['weights']['quant']

        align_size = 0
        shape = ()
        if type_ == 'ifm':
            shape = (self.batch, self.ic_size, self.ih_size, self.iw_size)
            ci_slice = 16
            if self.ifm_prec == "int10" or self.ifm_prec == 'int16':
                ci_slice = 8
            align_size = self.batch * _align_shape(int(self.ic_size), ci_slice) * self.ih_size * _align_shape(
                int(self.iw_size), ifm_iw) * self.get_dtype_size(self.ifm_prec)
        elif type_ == 'ofm' or type_ == 'elt':
            shape = (
                self.batch, self.oc_size, self.oh_size, self.ow_size)
            self.ofm_align_shape = (
                self.batch, _align_shape(int(self.oc_size), ofm_oc), self.oh_size,
                _align_shape(int(self.ow_size), ofm_ow))
            align_size = np.prod(self.ofm_align_shape) * self.get_dtype_size(self.ofm_prec)
        elif type_ == 'weight':
            ci_slice = 16  # TODO only support int8/10/16
            if self.ifm_prec == "int10" or self.ifm_prec == 'int16':
                ci_slice = 8
            shape = (self.oc_size, self.ic_size, self.kernel, self.kernel)
            kernel_align_shape = [ceil(self.oc_size / ofm_oc), ceil(self.ic_size / ci_slice), self.kernel, self.kernel,
                                  weights_kernel_oc, ci_slice]
            align_size = np.prod(kernel_align_shape)
        elif type_ == 'bias':
            shape = (self.oc_size,)
            align_size = _align_shape(int(self.oc_size), ofm_oc) * int(weights_bias / 8)
        elif type_ == 'quant' or type_ == 'dequant':
            shape = (self.oc_size,)
            align_size = _align_shape(int(self.oc_size), ofm_oc) * int(weights_quant / 8)

        return shape, align_size

    def gen_scalar_stmt(self, fused, content, veu_op_data=''):
        l_idx = content.find('(')
        r_idx = content.find(')')
        self.param = content[l_idx + 1: r_idx].split(',')
        print("fused param:", self.param)
        type_index = 0
        for s_name in self.param:
            if 'addr' in s_name:
                pass
            elif 'type' in s_name:
                s_value = veu_op_data["data_type"].split("+")[type_index]
                type_index += 1
            elif 'op_v' in s_name:
                s_value = veu_op_data["type_ins"]  # OP_MAP[fused]
            elif 'loop' in s_name:
                s_value = (math.prod(self.ofm_align_shape) + 127) // 128
            else:
                pass
            self.scalar_vec.append(Scalar(s_name.strip(), s_value).gen_scalar())
        logging_func()
        print('self.scalar_vec', self.scalar_vec)

    def get_buf_sequence(self):
        if "conv2d" in self.operator:
            self.buf_sequence = ['ifm', 'weight']
            if self.bias:
                self.buf_sequence.append('bias')
            if self.dqt:
                self.buf_sequence.append('quant')
            if self.elt:
                self.buf_sequence.append('elt')
                if self.elt_prec != 'i32':
                    self.buf_sequence.append('dequant')
            self.buf_sequence.append('ofm')
        return self.buf_sequence

    def generate_buffer(self):
        # L3 buffer
        addr = '0x0000'
        buf_sequence = ['ifm', 'ofm'] + self.buf_sequence[1: -1]
        ALIGN_TYPE_MAP['ifm'] = IFM_SHAPE[self.ifm_prec]
        for type_ in buf_sequence:
            if (type_ == 'weight'):
                self.paramAddr = addr
            if (type_ == 'ofm'):
                self.ofmAddr = addr
            align_size = self.cal_align_size(type_)[1]
            self.buf_vec.append(Buffer(L3_BUFFER, type_, ALIGN_TYPE_MAP[type_], str(self.cal_align_size(type_)[0]),
                                       self.prec_map[type_], addr, hex(align_size)).gen_buffer())
            addr = hex(int(addr, 16) + align_size)
        self.buf_vec.append("")

        # L2&L1 buffer
        for buffer in [L2_BUFFER, L1_BUFFER]:
            addr = '0x0000'
            for type_ in self.buf_sequence:
                align_size = self.cal_align_size(type_)[1]
                self.buf_vec.append(Buffer(buffer, type_, ALIGN_TYPE_MAP[type_], str(self.cal_align_size(type_)[0]),
                                           self.prec_map[type_], addr, hex(align_size)).gen_buffer())
                addr = hex(int(addr, 16) + align_size)
            self.buf_vec.append("")

    def gen_fused_op(self):
        print(self.veu_op)
        veu_op_file_data = ''
        if not self.fused[0]:
            asm_file = '/fused_avg_pooling.asm'
        else:
            print(FUSED_VEU_OP[self.raw_fused]["func"])
            veu_op_file_data = FUSED_VEU_OP[self.raw_fused]
            asm_file = '/' + veu_op_file_data["func"] + '.asm'
        # ---------
        # if fused exist
        # ---------
        print('asm_file:', asm_file)
        with open(env_config['veu_op_asm_path'] + asm_file, 'r') as f:
            content = f.readline().rstrip()
            while True:
                if 'loop_times) {' in content:
                    break
                content = f.readline().rstrip()

            self.fused_op = content.split('{')[0].strip()
            print(self.fused, content)
            self.gen_scalar_stmt(self.fused[0], content, veu_op_file_data)
            while content:
                self.fused_vec.append(content)
                content = f.readline().rstrip()

    def gen_sync_vec(self):
        idx = 0
        for _ in range(len(self.buf_sequence) - 1):
            self.sync_vec.append(Sync(idx, "cdtu.ld", "ndtu0.ld").gen_sync())
            idx += 1

        self.sync_vec.append(Sync(idx, "ndtu0.ld", "npu0").gen_sync())
        self.sync_vec.append(Sync(idx + 1, "npu0", "ndtu0.st").gen_sync())
        self.sync_vec.append(Sync(idx + 2, "ndtu0.st", "cdtu.st").gen_sync())
        return self.sync_vec

    def gen_conv_inst(self):
        stmt = ''
        if self.operator == "conv2d":
            stmt = f"conv.{self.padding[0]}.{self.padding[1]}.{self.padding[2]}.{self.padding[3]}.{self.stride}"
        elif self.operator == "winograd":
            stmt = f"winograd.{self.padding[0]}.{self.padding[1]}.{self.padding[2]}.{self.padding[3]}.{self.stride}"
        sub_buf = 'ofm_l1, '
        if self.bias:
            stmt += '.bias_en'
            sub_buf += 'bias_l1, '
        stmt += '.qt_en'
        sub_buf += 'quant_l1, '
        if self.elt:
            stmt += '.dqt_elt'
            if self.elt_prec != 'i32':
                sub_buf += 'dequant_l1, '
            sub_buf += 'elt_l1, '
        stmt += ' '
        stmt += sub_buf + 'weight_l1, ifm_l1;'
        return stmt

    def gen_cdtu_stream(self):
        self.cdtu_stream.append("cdtu_stream () {")
        for idx in range(len(self.buf_sequence) - 1):
            self.cdtu_stream.append(f"copy {self.buf_sequence[idx]}_l2, {self.buf_sequence[idx]}_l3;")
            self.cdtu_stream.append(f"send message{idx};")

        self.cdtu_stream.append('')
        self.cdtu_stream.append(f"wait message{len(self.buf_sequence) + 1};")
        self.cdtu_stream.append(f"copy output, {self.buf_sequence[-1]}_l2;")
        self.cdtu_stream.append("}")

    def gen_npu_stream(self):
        self.npu_stream.append("npu_stream () {")
        for idx in range(len(self.buf_sequence) - 1):
            self.npu_stream.append(f"wait message{idx};")
            self.npu_stream.append(f"copy {self.buf_sequence[idx]}_l1, {self.buf_sequence[idx]}_l2;")
        self.npu_stream.append(f'send message{len(self.buf_sequence) - 1};')

        self.npu_stream.append('')
        self.npu_stream.append(f'wait message{len(self.buf_sequence) - 1};')
        self.npu_stream.append(self.gen_conv_inst())
        # if False:

        if self.fused:
            self.npu_stream.append('fused.' + self.fused_op + ';')
        else:
            self.npu_stream.append('fused.fused_avg_pooling(loop_times);')
        self.npu_stream.append(f'send message{len(self.buf_sequence)};')

        self.npu_stream.append('')
        self.npu_stream.append(f'wait message{len(self.buf_sequence)};')
        self.npu_stream.append(f"copy {self.buf_sequence[-1]}_l2, {self.buf_sequence[-1]}_l1;")
        self.npu_stream.append(f"send message{len(self.buf_sequence) + 1};")
        self.npu_stream.append("}")

    def gen_submodule(self):
        logging_func()
        print('fused:', self.fused_op)
        self.get_buf_sequence()
        self.generate_buffer()
        self.gen_fused_op()
        self.config_vec = Config(1, self.paramAddr, self.ofmAddr).gen_config()
        self.entry_vec = Entry("cdtu", "npu(0)", self.veu_op).gen_entry()
        self.gen_sync_vec()
        self.gen_cdtu_stream()
        self.gen_npu_stream()

    def gen_asm_file(self, asm_name):
        with open(self.base_dir + "/" + asm_name + '.asm', 'w+') as f:
            f.write(VERSION_INFO)

            # wirte config
            for module in [self.config_vec, self.entry_vec, self.scalar_vec, self.buf_vec, self.sync_vec,
                           self.cdtu_stream, self.npu_stream, self.fused_vec]:
                if module:
                    for stmt in module:
                        f.write(stmt + '\n')
                    f.write("\n")

    def gen_conv_metadata(self):
        jsontext = {}
        if "conv2d" in self.operator:
            jsontext['op'] = "conv2d"
        else:
            jsontext["op"] = self.operator
        jsontext['fused'] = self.fused
        jsontext['bias_term'] = True
        jsontext['winograd'] = self.winograd
        jsontext['padding'] = self.padding
        jsontext['stride'] = [self.stride, self.stride]
        jsontext["input_params"] = self.buf_sequence

        print(self.buf_sequence)

        ifm_info = {}
        ifm_info['name'] = "ifm_0"
        ifm_info['fmt'] = 'c16w32'
        ifm_info['shape'] = [self.batch, self.ic_size, self.ih_size, self.iw_size]
        ifm_info['dtype'] = self.ifm_prec
        jsontext['ifm'] = ifm_info

        kernel_info = {}
        kernel_info['name'] = "weight_0"
        kernel_info['fmt'] = 'kernel'
        kernel_info['shape'] = [self.oc_size, self.ic_size, self.kernel, self.kernel]
        kernel_info['dtype'] = self.ifm_prec
        jsontext['weight'] = kernel_info

        bias_info = {}
        bias_info['name'] = "bias_0"
        bias_info['fmt'] = 'bias'
        bias_info['shape'] = [self.oc_size, ]
        bias_info['dtype'] = 'i32'
        jsontext['bias'] = bias_info

        quant_info = {}
        quant_info['name'] = "quant_0"
        quant_info['fmt'] = 'quant'
        quant_info['shape'] = [self.oc_size, 2]
        quant_info['dtype'] = 'i32'
        jsontext['quant'] = quant_info

        # TODO
        # elt_info = {}
        # elt_info['name'] = "elt_0"
        # elt_info['fmt'] = 'elt'
        # elt_info['shape'] = [self.oc_size, 2]
        # elt_info['dtype'] = 'i32'
        # jsontext['elt'] = elt_info
        #
        # dequant_info = {}
        # dequant_info['name'] = "dequant_0"
        # dequant_info['fmt'] = 'dequant'
        # dequant_info['shape'] = [self.oc_size, 2]
        # dequant_info['dtype'] = 'i32'
        # jsontext['dequant'] = dequant_info

        ofm_info = {}
        ofm_info['name'] = "ofm_0"
        ofm_info['fmt'] = 'c32w32'
        ofm_info['shape'] = [self.batch, self.oc_size, self.oh_size, self.ow_size]
        ofm_info['dtype'] = self.ofm_prec
        jsontext['ofm'] = ofm_info

        json_abs_path = os.path.abspath(self.base_dir + '/' + self.create_op_dir() + "_metadata.json")
        json.dump(jsontext, open(json_abs_path, "w"), indent=4)
        GenConvParamsData(json_abs_path, self.base_dir, self.create_op_dir()).run()


def get_random_testcase(dataf, case_num=3):
    for row_idx, ctx in dataf.iterrows():
        if row_idx == 2:
            print(ctx)
            size_vector = random_size(ctx, case_num)
            ifm_precision_list = ['i8']
            ofm_precision_list = ['i8']
            ifm_precision_ctx = ctx['ifm_prec'].strip()
            ofm_precision_ctx = ctx['ofm_prec'].strip()

            if ifm_precision_ctx:
                if ifm_precision_ctx == 'multiple':
                    ifm_precision_list = precion['ifm']  # , 'i16'
                else:
                    ifm_precision_list = ifm_precision_ctx.split('&')

            if ofm_precision_ctx:
                if ofm_precision_ctx == 'multiple':
                    ofm_precision_list = precion['ofm']
                else:
                    ofm_precision_list = ofm_precision_ctx.split('&')

            for item_size in size_vector:
                for ifm_prec in ifm_precision_list:
                    for ofm_rec in ofm_precision_list:
                        print(item_size, ifm_prec, ofm_rec)
                        conv_tcs = ConvTestcase(ctx)
                        conv_tcs.handle_attrs(ifm_prec, ofm_rec, item_size)
                        asm_name = conv_tcs.create_op_dir()
                        conv_tcs.gen_submodule()
                        conv_tcs.gen_asm_file(asm_name)

                        conv_tcs.gen_conv_metadata()

            break

    return 1


def get_fixed_testcase(file):
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info("start to generate asm!")
    print(sys.argv)
    # file = sys.argv[1]
    file = r'assembler_testcases_demo.xlsx'

    if not env_init(file):
        sys.exit()
    logging.info("env init done!")

    df = pd.read_excel(file, sheet_name="Sheet1")
    df.fillna('', inplace=True)

    # check data format
    for idx, content in df.iterrows():
        if case_format_check(idx, content.items()):
            # case error. bad format/value
            break
    logging.info("case format is right!")

    get_random_testcase(df, case_num=1)
    # get_fixed_testcase(file)
