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

def compute_softmax3D(list):
    arr = np.array(list)
    result_arr = np.zeros(arr.shape)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            result_arr[:,i,j] = compute_softmax(arr[:,i,j])
    return result_arr.tolist()

def compute_softmax(list):
    return np.exp(list) / sum(np.exp(list))

def compute_sigmoid(list):
    return 1 / (1 + np.exp(-np.array(list)))

def compute_mean(list, axis=None):
    return np.mean(list, axis)

def compute_variance(list, axis=None):
    return np.var(list, axis, ddof=1)

def compute_argmax(list, axis=None):
    return np.argmax(list, axis)

def compute_sum(list, axis=None):
    return np.sum(list, axis)

def compute_entropy(list):

    np_data = np.array(list)
    np_data = np.clip(np_data, 1e-7, 1)
    prob_logs = -np_data * np.log(np_data)
    entropy = np.sum(prob_logs)
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


def process_logits(logits):
    if len(compute_shape(logits)) == 1: # binary classifier
        positive_confidence = compute_sigmoid(logits).tolist()
        confidences = [positive_confidence[0], 1 - positive_confidence[0]]
        return confidences, None, None, None
    elif len(compute_shape(logits)) == 2: # multiple class classifier
        confidences = compute_softmax(logits).tolist()
        return confidences, None, None, None
    elif len(compute_shape(logits)) == 3: #semantic segmentation
        # dimension 0 is number of classes
        # dimension 1 is height
        # dimension 2 is width
        probability_masks = compute_softmax3D(logits)
        entropy = [compute_entropy(mask) for mask in probability_masks]
        # print(entropy)
        segmentation_class_mask = compute_argmax(logits, axis=0).tolist()
        # compute entropy
        entropy = compute_mean(entropy)
        confidences = [0 for _ in range(0, len(logits))]
        # for each class in the segmentation_class_mask, compute the confidence of the class based on the pixels in the mask
        # assign the confidence of all classes not in that mask to 0
        for i in range(0, len(logits)):
            # iterates through list of lists to check if i is in any of the lists making up the class mask
            if any(i in mask for mask in segmentation_class_mask):
                # find the pixels in the mask that equal i
                # compute the mean of the probabilities of those pixels
                # assign the confidence of that class to that mean
                confidences[i] = compute_mean([probability_masks[i][j][k] for j in range(0, len(logits[0])) for k in range(0, len(logits[0][0])) if segmentation_class_mask[j][k] == i])
        return confidences, segmentation_class_mask, entropy, None
    elif len(compute_shape(logits)) == 4: # semantic segmentation with dropout
        # dimension 0 is number of inferences
        # dimension 1 is number of classes
        # probability_masks is a list of probability masks for each inference
        probability_masks = []
        for i in range(0, len(logits)):
            probability_i = compute_softmax3D(logits[i])
            probability_masks.append(probability_i)
        # probability_means is the mean probabilities for each class for each inference over the image
        probability_means = compute_mean(probability_masks, axis = (2,3))     
        # variances is the variance of the probabilities for each class for each inference over the image
        variance = compute_mean(compute_variance(probability_means, axis = 0).tolist())
        # probabilities is the average probability for each class over all inferences
        # it is now 3 dimensional [num_classes, height, width]
        probabilities = compute_mean(probability_masks, axis = 0)
        segmentation_class_mask = compute_argmax(probabilities, axis=0).tolist()
        # compute entropy
        entropy = [compute_entropy(mask) for mask in probabilities]
        entropy = compute_mean(entropy)
        confidences = [0 for _ in range(0, len(logits[0]))]
        for i in range(0, len(logits[0])):
            if any(i in mask for mask in segmentation_class_mask):
                # find the pixels in the mask that equal i
                # compute the mean of the probabilities of those pixels
                # assign the confidence of that class to that mean
                confidences[i] = compute_mean([probabilities[i][j][k] for j in range(0, len(logits[0][0])) for k in range(0, len(logits[0][0][0])) if segmentation_class_mask[j][k] == i])

        return confidences, segmentation_class_mask, entropy, variance