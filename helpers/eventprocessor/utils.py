import os
import base64
import json
import math
from io import BytesIO
from copy import deepcopy
import lz4.frame
import numpy as np
from numpy import dot
from numpy.linalg import norm
from .pooling import pool2D


def encode_np_array(np_array, pool=False, flatten=False):

    np_array = np.array(np_array)
    if pool:
        np_array = pool_if_too_big(np_array)
    if flatten:
        np_array = flatten_if_not_flat(np_array)

    bytes_buffer = BytesIO()
    np.save(bytes_buffer, np_array.astype(dtype=np.float16))

    return base64.b64encode(
        lz4.frame.compress(
            bytes_buffer.getvalue(),
            compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
        )).decode('ascii')

def in_place_walk_decode_embeddings(my_dict):
    for key, value in my_dict.items():
        if isinstance(value, dict):
            in_place_walk_decode_embeddings(value)
        else:
            if key == 'embeddings' and isinstance(value, str):
                my_dict[key] = decode_np_array(value)

def decode_np_array(string_embedding, dtype=np.float16):
    decoded_bytes = lz4.frame.decompress(base64.b64decode(string_embedding))

    if decoded_bytes[:6] == b'\x93NUMPY':
        bytes_buffer = BytesIO(decoded_bytes)
        loaded_np = np.load(
            bytes_buffer, allow_pickle=True).astype(dtype=dtype)
        return loaded_np

    return np.frombuffer(decoded_bytes, dtype=dtype)

def pool_if_too_big(np_array):

    max_emb_size = int(os.environ.get('MAX_EMBEDDINGS_SIZE', 5000))

    np_array = np.array(np_array)
    if len(np_array.shape) == 3:
        total_weights = np_array.shape[0] * np_array.shape[1] * np_array.shape[2]
        if total_weights > max_emb_size:
            ksize_y = max(1, math.ceil(np_array.shape[0] /
                math.sqrt(max_emb_size / np_array.shape[2]) * np_array.shape[0] / np_array.shape[1]))
            ksize_x = max(1, math.ceil(np_array.shape[1] /
                math.sqrt(max_emb_size / np_array.shape[2]) * np_array.shape[1] / np_array.shape[0]))
            np_array = pool2D(np_array, (ksize_y, ksize_x), (ksize_y, ksize_x))
    return np_array

def flatten_if_not_flat(np_array):
    np_array = np.array(np_array)
    if len(np_array.shape) != 1:
        return np_array.flatten()
    return np_array

def compute_iou(bbox_1, bbox_2):

    max_left = max(bbox_1['left'], bbox_2['left'])
    max_top = max(bbox_1['top'], bbox_2['top'])
    min_right = min(bbox_1['left'] + bbox_1['width'], bbox_2['left'] + bbox_2['width'])
    min_bottom = min(bbox_1['top'] + bbox_1['height'], bbox_2['top'] + bbox_2['height'])

    intersection = max(0, min_right - max_left) * max(0, min_bottom - max_top)

    bbox_1_area = bbox_1['width'] * bbox_1['height']
    bbox_2_area = bbox_2['width'] * bbox_2['height']

    iou = intersection / float(bbox_1_area + bbox_2_area - intersection)
    return iou

def compute_cosine_similarity(list1, list2):
    result = dot(list1, list2)/(norm(list1)*norm(list2))
    return result

def compute_softmax(list):
    return np.exp(list) / sum(np.exp(list))

def compute_argmax(list):
    return np.argmax(list)

def compute_entropy(list):

    np_data = np.array(list)
    prob_logs = np_data * np.log2(np_data)
    numerator = 0 - np.sum(prob_logs)
    denominator = np.log2(np_data.shape[0])
    entropy = numerator / denominator
    return entropy

def compute_margin_of_confidence(list):
    new_list = deepcopy(list)
    new_list.sort(reverse=True)
    return 1 - (new_list[0] - new_list[1]) # to be fixed

def compute_ratio_of_confidence(list):
    new_list = deepcopy(list)
    new_list.sort(reverse=True)
    if new_list[1] != 0:
        return new_list[0] / new_list[1]
    return -1

class NpEncoder(json.JSONEncoder):
    def np_encoder_default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
