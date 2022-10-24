import copy
from itertools import zip_longest
import pandas as pd
from sklearn.metrics import ndcg_score

import numpy as np

from .utils import (
    encode_np_array,
    compute_iou,
    compute_cosine_similarity
)

from .stanford_squad import (
    compute_exact,
    compute_f1,
    get_tokens
)

from .pooling import (
    roi_pooling
)

from jiwer import compute_measures

def preprocess_generic(event):
    gt_bboxes = event.get('groundtruth', [])
    pred_bboxes = event.get('prediction', [])

    matches = []
    for (g, p) in zip_longest(gt_bboxes, pred_bboxes):
        event_copy = copy.deepcopy(event)
        event_copy.pop('embeddings', None)
        matches.append({
            **event_copy,
            'prediction': p,
            'groundtruth': g
        })
    
    return matches

def preprocess_object_detection(json_event):
    gt_bboxes = json_event.get('groundtruth', [])
    pred_bboxes = json_event.get('prediction', [])

    image_embeddings = json_event.get('embeddings', None)
    image_metadata = json_event.get('image_metadata', None)
    image_height = image_metadata.get('height', None) if image_metadata else None
    image_width = image_metadata.get('width', None) if image_metadata else None
    pred_box_embeddings = None
    gt_box_embeddings = None

    # Generate prediction and groundtruth embeddings.
    # TODO: Use RoIPooling instead of this (see metrics-engine/roi_generator.py)
    if image_embeddings and image_height and image_width:
        if len(pred_bboxes) > 0:
            if 'embeddings' not in pred_bboxes[0]:
                np_pred_box = np.zeros((len(pred_bboxes), 4))
                for index, pred_bbox in enumerate(pred_bboxes):
                    np_pred_box[index][0] = pred_bbox['top'] / image_height
                    np_pred_box[index][1] = pred_bbox['left'] / image_width
                    np_pred_box[index][2] = pred_bbox['height'] / image_height
                    np_pred_box[index][3] = pred_bbox['width'] / image_width

                pred_box_embeddings = roi_pooling(image_embeddings, np_pred_box, 5)
        if len(gt_bboxes) > 0:
            if 'embeddings' not in gt_bboxes[0]:
                np_gt_box = np.zeros((len(gt_bboxes), 4))
                for index, gt_bbox in enumerate(gt_bboxes):
                    np_gt_box[index][0] = gt_bbox['top'] / image_height
                    np_gt_box[index][1] = gt_bbox['left'] / image_width
                    np_gt_box[index][2] = gt_bbox['height'] / image_height
                    np_gt_box[index][3] = gt_bbox['width'] / image_width

                gt_box_embeddings = roi_pooling(image_embeddings, np_gt_box, 5)

    for index, bbox in enumerate(gt_bboxes):
        bbox['id'] = index

    for index, bbox in enumerate(pred_bboxes):
        bbox['id'] = index

    gt_no_match = list(range(len(gt_bboxes)))
    pred_no_match = list(range(len(pred_bboxes)))

    json_event_copy = copy.deepcopy(json_event)
    json_event_copy.pop('embeddings', None)

    matches = []

    for gt_bbox in gt_bboxes:
        best_iou = 0
        best_pred = None
        for pred_bbox in pred_bboxes:
            iou = compute_iou(gt_bbox, pred_bbox)
            if iou >= 0.5 and iou > best_iou:
                best_iou = iou
                best_pred = pred_bbox

        if best_pred is not None:
            if gt_bbox['id'] in gt_no_match:
                gt_no_match.remove(gt_bbox['id'])
            if best_pred['id'] in pred_no_match:
                pred_no_match.remove(best_pred['id'])

            copy_event = copy.deepcopy(json_event_copy)
            copy_event['metrics'] = copy_event.get('metrics', {})
            copy_event['metrics']['iou'] = best_iou

            if pred_box_embeddings is not None:
                copy_event['prediction'] = copy_event.get('prediction', {})
                copy_event['prediction']['embeddings'] = encode_np_array(pred_box_embeddings[best_pred['id']], flatten=True)

            if gt_box_embeddings is not None:
                copy_event['groundtruth'] = copy_event.get('groundtruth', {})
                copy_event['groundtruth']['embeddings'] = encode_np_array(gt_box_embeddings[gt_bbox['id']], flatten=True)

            matches.append(copy_event)

    for no_match_index in gt_no_match:
        gt_bbox = gt_bboxes[no_match_index]
        copy_event = copy.deepcopy(json_event_copy)
        if gt_box_embeddings is not None:
                copy_event['groundtruth'] = copy_event.get('groundtruth', {})
                copy_event['groundtruth']['embeddings'] = encode_np_array(gt_box_embeddings[gt_bbox['id']], flatten=True)

        matches.append(copy_event)

    for no_match_index in pred_no_match:
        pred_bbox = pred_bboxes[no_match_index]
        copy_event = copy.deepcopy(json_event_copy)
        if pred_box_embeddings is not None:
                copy_event['prediction'] = copy_event.get('prediction', {})
                copy_event['prediction']['embeddings'] = encode_np_array(pred_box_embeddings[best_pred['id']], flatten=True)

        matches.append(copy_event)

    return matches

def preprocess_learning_to_rank(json_event):
    gt_bboxes = json_event.get('groundtruth', [])
    pred_bboxes = json_event.get('prediction', [])
    relevance = pd.DataFrame(gt_bboxes)['relevance']
    score = pd.DataFrame(pred_bboxes)['score']
    relevance_score = pd.concat([relevance, score], axis=1)
    relevance_score = relevance_score.sort_values(by=['score'], ascending=False)

    json_event['metrics'] = json_event.get('metrics', {})
    json_event['metrics']['ndcg'] = ndcg_score(
        [relevance.sort_values(ascending=False).to_numpy()], 
        [relevance_score['relevance'].to_numpy()]
    )
    json_event['metrics']['rr'] = sum([
        r_s['relevance'] / (i + 1) for i, r_s in relevance_score.iterrows() 
    ])

    features = json_event.pop('features', None)

    matches = []
    for i, (g, p) in enumerate(zip_longest(gt_bboxes, pred_bboxes)):
        event_copy = copy.deepcopy(json_event)
        event_copy.pop('embeddings', None)
        matches.append({
            **event_copy,
            'prediction': p,
            'groundtruth': g,
            'features': features[i] if features else None,
            'text': json_event['text']['documents'][i] if 'text' in json_event and 'documents' in json_event['text'] else None,
        })
    
    if 'text' in json_event and 'query' in json_event['text']:
        json_event['text'] = json_event['text']['query']
    
    return matches

def preprocess_question_answering(json_event):
    gt_answer = json_event.pop('groundtruth')
    pred_answers = json_event.pop('prediction')

    gt_text = gt_answer['text']

    best_answer = {}
    best_f1 = 0.0
    best_extact_match = 0.0

    for pred_answer in pred_answers:
        pred_text = pred_answer['text']
        exact_match = compute_exact(gt_text, pred_text)

        if bool(exact_match):
            best_answer = pred_answer
            best_f1 = 1.0
            best_extact_match = 1.0
            break

        f1_match = compute_f1(gt_text, pred_text)

        if not best_answer or f1_match > best_f1:
            best_answer = pred_answer
            best_f1 = f1_match

    gt_embeddings_vector = []
    pred_embeddings_vector = []
    if 'embeddings' in gt_answer:
        embeddings = gt_answer.pop('embeddings')
        gt_embeddings_vector = embeddings
        gt_answer['embeddings'] = encode_np_array(embeddings, flatten=True)

    if 'embeddings' in best_answer:
        embeddings = best_answer.pop('embeddings')
        pred_embeddings_vector = embeddings
        best_answer['embeddings'] = encode_np_array(embeddings, flatten=True)

    if (gt_embeddings_vector and pred_embeddings_vector):
        cosine_similarity = compute_cosine_similarity(gt_embeddings_vector, pred_embeddings_vector)

    json_event['f1_score'] = best_f1
    json_event['exact_match'] = best_extact_match

    if (gt_embeddings_vector and pred_embeddings_vector):
        json_event['semantic_similarity'] = cosine_similarity

    return [json_event]

def preprocess_automated_speech_recognition(json_event):
    processed_event = copy.deepcopy(json_event)
    processed_event.pop('embeddings', None)
    groundtruth = processed_event.get('groundtruth')
    prediction = processed_event.get('prediction')

    measures = compute_measures(groundtruth['text'], prediction['text'])

    prediction['wer'] = measures['wer']
    prediction['substitutions'] = measures['substitutions']
    prediction['deletions'] = measures['deletions']
    prediction['insertions'] = measures['insertions']
    prediction['hits'] = measures['hits']

    if measures['wer'] == 0:
        prediction['exact_match'] = 1
    else:
        prediction['exact_match'] = 0

    return [processed_event]

def preprocess_auto_completion(json_event):

    gt_answer = json_event.pop('groundtruth')
    pred_answers = json_event.pop('prediction')
    gt_text = gt_answer['text']
    # TODO: Taking only the top pred for now
    pred_text = pred_answers[0]['text']

    best_f1 = 0.0
    best_extact_match = 0.0
    best_match_token_length = 0.0

    pred_text_tokens = get_tokens(pred_text, normalize=False)
    truncated_gt_tokens = get_tokens(gt_text, normalize=False)

    if pred_text_tokens == truncated_gt_tokens[0: len(pred_text_tokens)]:
        best_f1 = 1.0
        best_extact_match = 1.0
        best_match_token_length = len(truncated_gt_tokens)
    else:

        for token_index in range(
                max(0, -4 + len(pred_text_tokens)),
                min(len(truncated_gt_tokens), 4 + len(pred_text_tokens))):
            cropped_gt = ' '.join(truncated_gt_tokens[0: token_index])

            my_f1 = compute_f1(pred_text, cropped_gt, normalize=False)

            if my_f1 > best_f1:
                best_f1 = my_f1
                best_match_token_length = token_index

    if 'embeddings' in gt_answer:
        gt_answer.pop('embeddings')

    json_event['f1_score'] = best_f1
    json_event['exact_match'] = best_extact_match
    json_event['best_match_token_length'] = best_match_token_length

    return [json_event]

def preprocess_semantic_similarity(json_event):
    input_data = json_event.pop('input_data')
    input_embeddings_1 = input_data['embeddings_1']
    input_embeddings_2 = input_data['embeddings_2']

    encoded_embedding_1 = encode_np_array(input_embeddings_1)

    input_data['embeddings_1'] = encoded_embedding_1
    input_data['embeddings_2'] = encode_np_array(input_embeddings_2)

    json_event['embeddings'] = encoded_embedding_1

    json_event['cosine_score'] = compute_cosine_similarity(
        input_embeddings_1, input_embeddings_2)

    return [json_event]
