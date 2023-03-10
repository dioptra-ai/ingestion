import os
import base64
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

    if pool and len(np_array.shape) == 3:
        max_emb_size = int(os.environ.get('MAX_EMBEDDINGS_SIZE', 5000))
        total_weights = np_array.shape[0] * np_array.shape[1] * np_array.shape[2]

        if total_weights > max_emb_size:
            ksize_y = max(1, math.ceil(np_array.shape[0] /
                math.sqrt(max_emb_size / np_array.shape[2]) * np_array.shape[0] / np_array.shape[1]))
            ksize_x = max(1, math.ceil(np_array.shape[1] /
                math.sqrt(max_emb_size / np_array.shape[2]) * np_array.shape[1] / np_array.shape[0]))
            np_array = pool2D(np_array, (ksize_y, ksize_x), (ksize_y, ksize_x))

    if flatten and len(np_array.shape) != 1:
        np_array = np_array.flatten()

    bytes_buffer = BytesIO()
    np.save(bytes_buffer, np_array.astype(dtype=np.float16))

    return base64.b64encode(
        lz4.frame.compress(
            bytes_buffer.getvalue(),
            compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
        )
    ).decode('ascii')

def compute_shape(list):
    return np.array(list).shape

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

def compute_softmax2D(list):
    arr = np.array(list)
    result = compute_softmax(arr)
    return result.reshape(arr.shape).tolist()

def compute_softmax(list):
    return np.exp(list) / sum(np.exp(list))

def compute_sigmoid(list):
    return 1 / (1 + np.exp(-np.array(list)))

def compute_mean(list, axis=None):
    return np.mean(list, axis)

def compute_variance(list, axis=None):
    return np.var(list, axis)

def compute_argmax(list, axis=None):
    return np.argmax(list, axis)

def compute_sum(list, axis=None):
    return np.sum(list, axis)

def compute_entropy(list):

    np_data = np.array(list)
    np_data = np_data[np_data != 0] #filtering out 0 values
    prob_logs = np_data * np.log2(np_data)
    numerator = 0 - np.sum(prob_logs)
    denominator = np.log2(np_data.shape[0])
    if denominator == 0:
        return -1
    entropy = numerator / denominator
    if np.isnan(entropy):
        return -1
    return entropy

def compute_margin_of_confidence(list):
    new_list = deepcopy(list)
    new_list.sort(reverse=True)
    return new_list[0] - new_list[1]

def compute_ratio_of_confidence(list):
    new_list = deepcopy(list)
    new_list.sort(reverse=True)
    if new_list[1] != 0:
        return new_list[0] / new_list[1]
    return -1
