import os
import base64
import math
from io import BytesIO
from copy import deepcopy
import lz4.frame
import numpy as np
from numpy import dot
from numpy.linalg import norm
from PIL import Image
import orjson
from helpers.eventprocessor.pooling import pool2D

DIOPTRA_MASK_RESIZE = int(os.environ.get('DIOPTRA_MASK_RESIZE', 512))

def decode_to_np_array(value):
    if isinstance(value, str):
        decoded_bytes = lz4.frame.decompress(base64.b64decode(value))

        if decoded_bytes[:6] == b'\x93NUMPY':

            return np.load(BytesIO(decoded_bytes), allow_pickle=True).astype(dtype=np.float16)

        else:

            return np.frombuffer(decoded_bytes, dtype=np.float16)
    elif isinstance(value, dict):

        return np.array(list(value.values()), dtype=np.float32)
    else:
        raise Exception(f'Unknown type: {type(value)}')

def encode_np_array(np_array, pool=False, flatten=False):

    if not isinstance(np_array, np.ndarray):
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


def encode_list(my_list):

    if isinstance(my_list, np.ndarray):
        my_list = my_list.tolist()

    return base64.b64encode(
        lz4.frame.compress(
            orjson.dumps(my_list),
            compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
        )
    ).decode('ascii')

def decode_list(value):
    decoded_bytes = lz4.frame.decompress(base64.b64decode(value))
    return orjson.loads(decoded_bytes)

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

def squeeze(list):
    return np.squeeze(list)

def compute_cosine_similarity(list1, list2):
    result = dot(list1, list2)/(norm(list1)*norm(list2))
    return result

def compute_softmax(list, axis=None):
    return np.exp(list) / compute_sum(np.exp(list), axis, True)

def compute_sigmoid(list):
    return 1 / (1 + np.exp(-np.array(list)))

def compute_mean(list, axis=None):
    return np.mean(list, axis)

def compute_variance(list, axis=None):
    return np.var(list, axis)

def compute_argmax(list, axis=None):
    return np.argmax(list, axis)

def compute_max(list, axis=None):
    return np.max(list, axis)

def compute_sum(list, axis=None, keepdims=False):
    return np.sum(list, axis=axis, keepdims=keepdims)

def compute_entropy(list, axis=None):

    np_data = np.array(list)
    np_data = np.clip(np_data, 1e-10, 1)

    aggregation_axis = [0]
    if axis is not None:
        if isinstance(axis, int):
            aggregation_axis = (axis,)
        else:
            aggregation_axis = axis
    base = 1
    for my_axis in aggregation_axis:
        base *= np_data.shape[my_axis]

    prob_logs = -np_data * np.log(np_data)
    entropy = compute_sum(prob_logs, axis) / np.log(base)
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

def resize_mask(segmentation_class_mask,dtype=np.uint16):
    if not isinstance(segmentation_class_mask, np.ndarray):
        segmentation_class_mask = np.array(segmentation_class_mask)
    segmentation_class_mask = segmentation_class_mask.astype(dtype) # max 65535 classes

    my_img = Image.fromarray(segmentation_class_mask)
    my_img = my_img.resize((DIOPTRA_MASK_RESIZE, DIOPTRA_MASK_RESIZE), resample=Image.NEAREST)
    return np.array(my_img)


def process_logits(logits, class_names=None):
    if len(compute_shape(logits)) == 1 and len(logits) == 1: # binary classifier
        positive_confidence = compute_sigmoid(logits)
        confidences = [positive_confidence[0], 1 - positive_confidence[0]]
        max_index = compute_argmax(confidences)
        class_name = class_names[max_index] if class_names is not None else max_index
        entropy = compute_entropy(confidences)
    
        return {
            'confidences': confidences,
            'confidence': confidences[max_index],
            'class_name': str(class_name),
            'entropy': entropy
        }
    elif len(compute_shape(logits)) == 1 and len(logits) > 1: # multiple class classifier
        confidences = compute_softmax(logits)
        max_index = compute_argmax(confidences)
        class_name = class_names[max_index] if class_names is not None else max_index
        entropy = compute_entropy(confidences)

        return {
            'confidences': confidences.tolist(),
            'confidence': confidences[compute_argmax(confidences)],
            'class_name': str(class_name),
            'entropy': entropy
        }
    elif len(compute_shape(logits)) == 3: #semantic segmentation
        # dimension 0 is number of classes
        # dimension 1 is height
        # dimension 2 is width
        pixel_confidences = np.clip(compute_softmax(logits, axis=0), 1e-10, 1)
        pixel_entropy = compute_entropy(pixel_confidences, axis=0)
        segmentation_class_mask = compute_argmax(logits, axis=0)
        entropy = compute_mean(pixel_entropy)
        confidences = compute_mean(pixel_confidences, axis=(1, 2))
        return {
            'confidences': confidences.tolist(),
            'confidence': compute_mean(confidences),
            'entropy': entropy,
            'segmentation_class_mask': segmentation_class_mask.tolist(),
            'pixel_entropy': pixel_entropy.tolist()
        }
    elif len(compute_shape(logits)) == 4: # semantic segmentation with dropout
        # dimension 0 is number of inferences
        # dimension 1 is number of classes
        # dimension 2 is height
        # dimension 3 is width
        inference_pixel_confidences = np.clip(compute_softmax(logits, axis=1), 1e-10, 1)
        pixel_confidences = compute_mean(inference_pixel_confidences, axis=0)
        pixel_entropy = compute_entropy(pixel_confidences, axis=0)
        pixel_variance = compute_mean(compute_variance(inference_pixel_confidences, axis=0), axis=0)
        segmentation_class_mask = compute_argmax(pixel_confidences, axis=0)
        entropy = compute_mean(pixel_entropy)
        variance = compute_mean(pixel_variance)
        confidences = compute_mean(pixel_confidences, axis=(1, 2))

        return {
            'confidences': confidences.tolist(),
            'confidence': compute_mean(confidences),
            'entropy': entropy,
            'variance': variance,
            'segmentation_class_mask': segmentation_class_mask.tolist(),
            'pixel_entropy': pixel_entropy.tolist(),
            'pixel_variance': pixel_variance.tolist()
        }
    else:
        raise Exception('Unknown logits shape: {}'.format(compute_shape(logits)))
