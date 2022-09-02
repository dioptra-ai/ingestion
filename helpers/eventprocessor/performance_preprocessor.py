import orjson
import copy

import numpy as np

from .utils import (
    encode_np_array,
    compute_iou,
    compute_cosine_similarity,
    compute_softmax,
    compute_argmax,
    compute_entropy,
    compute_margin_of_confidence,
    compute_ratio_of_confidence
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

def preprocess_object_detection(json_event):

    raw_gt_bboxes = json_event.pop('groundtruth', None)
    raw_pred_bboxes = json_event.pop('prediction', None)

    pred_bboxes = []
    gt_bboxes = []

    if raw_gt_bboxes is not None:
        if isinstance(raw_gt_bboxes, dict):
            boxes = raw_gt_bboxes['boxes']
            class_names = raw_gt_bboxes['class_names']
            for index, bbox in enumerate(boxes):
                gt_bboxes.append({
                        'top': bbox[0],
                        'left': bbox[1],
                        'height': bbox[2],
                        'width': bbox[3],
                        'class_name': class_names[index],
                        **(
                            {
                                'embeddings': bbox['embeddings']
                            } if 'embeddings' in bbox else {}
                        )
                    })
        else:
            gt_bboxes = raw_gt_bboxes

    if raw_pred_bboxes is not None:
        if isinstance(raw_pred_bboxes, dict):
            boxes = raw_pred_bboxes['boxes']
            confidences = raw_pred_bboxes.get('confidences', None)
            objectness = raw_pred_bboxes.get('objectness', None)
            logits = raw_pred_bboxes.get('logits', None)
            embeddings = raw_pred_bboxes.get('embeddings', None)
            class_names = raw_pred_bboxes['class_names']
            for index, bbox in enumerate(boxes):
                box_confidences = None
                if confidences is not None:
                    box_confidences = confidences[index]
                else:
                    if logits is not None:
                        box_confidences = compute_softmax(logits[index]).tolist()
                box_confidence = None
                box_class_name = None
                entropy = None
                ratio_of_confidence = None
                margin_of_confidence = None
                if box_confidences is not None:
                    if isinstance(box_confidences, list):
                        max_index = compute_argmax(box_confidences)
                        box_confidence = box_confidences[max_index]
                        box_class_name = class_names[max_index]
                        entropy = compute_entropy(box_confidences)
                        ratio_of_confidence = compute_ratio_of_confidence(box_confidences)
                        margin_of_confidence = compute_margin_of_confidence(box_confidences)
                    else:
                        box_confidence = box_confidences
                        box_class_name = class_names[index]
                else:
                    box_class_name = class_names[index]

                pred_bboxes.append({
                        'top': bbox[0],
                        'left': bbox[1],
                        'height': bbox[2],
                        'width': bbox[3],
                        **(
                            {
                                'logits': encode_np_array(logits[index], flatten=True)
                            } if logits is not None else {}
                        ),
                        **(
                            {
                                'confidence': box_confidence
                            } if box_confidence is not None else {}
                        ),
                        **(
                            {
                                'class_name': box_class_name
                            } if box_class_name is not None else {}
                        ),
                        **(
                            {
                                'entropy': entropy
                            } if entropy is not None else {}
                        ),
                        **(
                            {
                                'ratio_of_confidence': ratio_of_confidence
                            } if ratio_of_confidence is not None else {}
                        ),
                        **(
                            {
                                'margin_of_confidence': margin_of_confidence
                            } if margin_of_confidence is not None else {}
                        ),
                        **(
                            {
                                'objectness': objectness[index]
                            } if objectness is not None else {}
                        ),
                        **(
                            {
                                'embeddings': encode_np_array(embeddings[index], flatten=True)
                            } if embeddings is not None else {}
                        )
                    })
        else:
            for bbox in raw_pred_bboxes:
                box_confidences = bbox.get('confidence', None)
                logits = bbox.get('logits', None)
                class_name = bbox['class_name']

                if logits is not None:
                    box_confidences = compute_softmax(logits).tolist()

                box_confidence = None
                box_class_name = None
                entropy = None
                ratio_of_confidence = None
                margin_of_confidence = None

                if box_confidences is not None:
                    if isinstance(box_confidences, list):
                        max_index = compute_argmax(box_confidences)
                        box_confidence = box_confidences[max_index]
                        box_class_name = class_name[max_index]
                        entropy = compute_entropy(box_confidences)
                        ratio_of_confidence = compute_ratio_of_confidence(box_confidences)
                        margin_of_confidence = compute_margin_of_confidence(box_confidences)
                    else:
                        box_confidence = box_confidences
                        box_class_name = class_name
                else:
                    box_confidence = box_confidences
                    box_class_name = class_name

                pred_bboxes.append({
                        'top': bbox['top'],
                        'left': bbox['left'],
                        'height': bbox['height'],
                        'width': bbox['width'],
                        **(
                            {
                                'logits': encode_np_array(logits, flatten=True)
                            } if logits is not None else {}
                        ),
                        **(
                            {
                                'confidence': box_confidence
                            } if box_confidence is not None else {}
                        ),
                        **(
                            {
                                'class_name': box_class_name
                            } if box_class_name is not None else {}
                        ),
                        **(
                            {
                                'entropy': entropy
                            } if entropy is not None else {}
                        ),
                        **(
                            {
                                'ratio_of_confidence': ratio_of_confidence
                            } if ratio_of_confidence is not None else {}
                        ),
                        **(
                            {
                                'margin_of_confidence': margin_of_confidence
                            } if margin_of_confidence is not None else {}
                        ),
                        **(
                            {
                                'objectness': bbox['objectness']
                            } if 'objectness' in bbox else {}
                        ),
                        **(
                            {
                                'embeddings': encode_np_array(bbox['embeddings'], flatten=True)
                            } if 'embeddings' in bbox else {}
                        )
                    })

    image_embeddings = json_event.pop('non_encoded_embeddings', None)
    image_height = json_event.get('image_metadata.height', None)
    image_width = json_event.get('image_metadata.height', None)
    pred_box_embeddings = None
    gt_box_embeddings = None

    if image_embeddings is not None and image_height and image_width:
        np_image_embeddings = np.array(image_embeddings)
        if len(np_image_embeddings.shape) == 3:
            if len(pred_bboxes) > 0:
                if 'embeddings' not in pred_bboxes[0]:
                    np_pred_box = np.zeros((len(pred_bboxes), 4))
                    for index, pred_bbox in enumerate(pred_bboxes):
                        np_pred_box[index][0] = pred_bbox['top'] / image_height
                        np_pred_box[index][1] = pred_bbox['left'] / image_width
                        np_pred_box[index][2] = pred_bbox['height'] / image_height
                        np_pred_box[index][3] = pred_bbox['width'] / image_width

                    pred_box_embeddings = roi_pooling(
                        np_image_embeddings, np_pred_box, 5)
            if len(gt_bboxes) > 0:
                if 'embeddings' not in gt_bboxes[0]:
                    np_gt_box = np.zeros((len(gt_bboxes), 4))
                    for index, gt_bbox in enumerate(gt_bboxes):
                        np_gt_box[index][0] = gt_bbox['top'] / image_height
                        np_gt_box[index][1] = gt_bbox['left'] / image_width
                        np_gt_box[index][2] = gt_bbox['height'] / image_height
                        np_gt_box[index][3] = gt_bbox['width'] / image_width

                    gt_box_embeddings = roi_pooling(
                        np_image_embeddings, np_gt_box, 5)

    for index, bbox in enumerate(gt_bboxes):
        bbox['id'] = index

    for index, bbox in enumerate(pred_bboxes):
        bbox['id'] = index

    gt_no_match = list(range(len(gt_bboxes)))
    pred_no_match = list(range(len(pred_bboxes)))

    json_event['is_bbox_row'] = False

    json_event_copy = copy.deepcopy(json_event)
    json_event_copy.pop('embeddings', None)
    json_event_copy.pop('non_encoded_embeddings', None)

    if image_embeddings is not None:
        json_event['original_embeddings'] = encode_np_array(image_embeddings)

    matches = [json_event]

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
            copy_event['iou'] = best_iou
            copy_event['is_bbox_row'] = True

            if pred_box_embeddings is not None:
                copy_event['prediction.embeddings'] = encode_np_array(
                    pred_box_embeddings[best_pred['id']], flatten=True)

            if gt_box_embeddings is not None:
                copy_event['groundtruth.embeddings'] = encode_np_array(
                    gt_box_embeddings[gt_bbox['id']], flatten=True)

            matches.append(copy_event)

    for no_match_index in gt_no_match:
        gt_bbox = gt_bboxes[no_match_index]
        copy_event = copy.deepcopy(json_event_copy)
        copy_event['is_bbox_row'] = True
        if gt_box_embeddings is not None:
            copy_event['groundtruth.embeddings'] = encode_np_array(
                gt_box_embeddings[gt_bbox['id']], flatten=True)
        matches.append(copy_event)

    for no_match_index in pred_no_match:
        pred_bbox = pred_bboxes[no_match_index]
        copy_event = copy.deepcopy(json_event_copy)
        copy_event['is_bbox_row'] = True
        if pred_box_embeddings is not None:
            copy_event['prediction.embeddings'] = encode_np_array(
                pred_box_embeddings[pred_bbox['id']], flatten=True)
        matches.append(copy_event)

    return matches


def preprocess_automated_speech_recognition(json_event):

    groundtruth = json_event.pop('groundtruth')
    prediction = json_event.pop('prediction')

    measures = compute_measures(groundtruth['text'], prediction['text'])

    json_event['wer'] = measures['wer']
    json_event['substitutions'] = measures['substitutions']
    json_event['deletions'] = measures['deletions']
    json_event['insertions'] = measures['insertions']
    json_event['hits'] = measures['hits']

    if measures['wer'] == 0:
        json_event['exact_match'] = 1
    else:
        json_event['exact_match'] = 0

    return [json_event]


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

    json_event['cosine_score'] = compute_cosine_similarity(input_embeddings_1, input_embeddings_2)

    return [json_event]


def preprocess_classifier(json_event):
    if not isinstance(json_event['prediction'], dict): # We can't do anything
        return [json_event]

    prediction = json_event.pop('prediction')

    if 'confidence' in prediction and 'class_name' in prediction:
        max_class_index = compute_argmax(prediction['confidence'])

        json_event['prediction'] = prediction['class_name'][max_class_index]
        json_event['confidence'] = prediction['confidence'][max_class_index]
        json_event['entropy'] = compute_entropy(prediction['confidence'])
        json_event['ratio_of_confidence'] = compute_ratio_of_confidence(prediction['confidence'])
        json_event['margin_of_confidence'] = compute_margin_of_confidence(prediction['confidence'])

    if 'logits' in prediction:
        json_event['logits'] = encode_np_array(prediction['logits'])

    json_event['prediction_string'] = orjson.dumps(prediction).decode('ascii')

    return [json_event]


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
        cosine_similarity = compute_cosine_similarity(
            gt_embeddings_vector, pred_embeddings_vector)

    json_event['f1_score'] = best_f1
    json_event['exact_match'] = best_extact_match

    if (gt_embeddings_vector and pred_embeddings_vector):
        json_event['semantic_similarity'] = cosine_similarity

    return [json_event]


def preprocess_object_detection(json_event):

    raw_gt_bboxes = json_event.pop('groundtruth', None)
    raw_pred_bboxes = json_event.pop('prediction', None)

    pred_bboxes = []
    gt_bboxes = []

    if raw_gt_bboxes is not None:
        if isinstance(raw_gt_bboxes, dict):
            boxes = raw_gt_bboxes['boxes']
            class_names = raw_gt_bboxes['class_names']
            for index, bbox in enumerate(boxes):
                gt_bboxes.append({
                    'top': bbox[0],
                    'left': bbox[1],
                    'height': bbox[2],
                    'width': bbox[3],
                    'class_name': class_names[index],
                    **(
                        {
                            'embeddings': bbox['embeddings']
                        } if 'embeddings' in bbox else {}
                    )
                })
        else:
            gt_bboxes = raw_gt_bboxes

    if raw_pred_bboxes is not None:
        if isinstance(raw_pred_bboxes, dict):
            boxes = raw_pred_bboxes['boxes']
            confidences = raw_pred_bboxes.get('confidences', None)
            objectness = raw_pred_bboxes.get('objectness', None)
            logits = raw_pred_bboxes.get('logits', None)
            embeddings = raw_pred_bboxes.get('embeddings', None)
            class_names = raw_pred_bboxes['class_names']
            for index, bbox in enumerate(boxes):
                box_confidences = None
                if confidences is not None:
                    box_confidences = confidences[index]
                else:
                    if logits is not None:
                        box_confidences = compute_softmax(
                            logits[index]).tolist()
                box_confidence = None
                box_class_name = None
                entropy = None
                ratio_of_confidence = None
                margin_of_confidence = None
                if box_confidences is not None:
                    if isinstance(box_confidences, list):
                        max_index = compute_argmax(box_confidences)
                        box_confidence = box_confidences[max_index]
                        box_class_name = class_names[max_index]
                        entropy = compute_entropy(box_confidences)
                        ratio_of_confidence = compute_ratio_of_confidence(
                            box_confidences)
                        margin_of_confidence = compute_margin_of_confidence(
                            box_confidences)
                    else:
                        box_confidence = box_confidences
                        box_class_name = class_names[index]
                else:
                    box_class_name = class_names[index]

                pred_bboxes.append({
                    'top': bbox[0],
                    'left': bbox[1],
                    'height': bbox[2],
                    'width': bbox[3],
                    **(
                        {
                            'logits': encode_np_array(logits[index], flatten=True)
                        } if logits is not None else {}
                    ),
                    **(
                        {
                            'confidence': box_confidence
                        } if box_confidence is not None else {}
                    ),
                    **(
                        {
                            'class_name': box_class_name
                        } if box_class_name is not None else {}
                    ),
                    **(
                        {
                            'entropy': entropy
                        } if entropy is not None else {}
                    ),
                    **(
                        {
                            'ratio_of_confidence': ratio_of_confidence
                        } if ratio_of_confidence is not None else {}
                    ),
                    **(
                        {
                            'margin_of_confidence': margin_of_confidence
                        } if margin_of_confidence is not None else {}
                    ),
                    **(
                        {
                            'objectness': objectness[index]
                        } if objectness is not None else {}
                    ),
                    **(
                        {
                            'embeddings': encode_np_array(embeddings[index], flatten=True)
                        } if embeddings is not None else {}
                    )
                })
        else:
            for bbox in raw_pred_bboxes:
                box_confidences = bbox.get('confidence', None)
                logits = bbox.get('logits', None)
                class_name = bbox['class_name']

                if logits is not None:
                    box_confidences = compute_softmax(logits).tolist()

                box_confidence = None
                box_class_name = None
                entropy = None
                ratio_of_confidence = None
                margin_of_confidence = None

                if box_confidences is not None:
                    if isinstance(box_confidences, list):
                        max_index = compute_argmax(box_confidences)
                        box_confidence = box_confidences[max_index]
                        box_class_name = class_name[max_index]
                        entropy = compute_entropy(box_confidences)
                        ratio_of_confidence = compute_ratio_of_confidence(
                            box_confidences)
                        margin_of_confidence = compute_margin_of_confidence(
                            box_confidences)
                    else:
                        box_confidence = box_confidences
                        box_class_name = class_name
                else:
                    box_confidence = box_confidences
                    box_class_name = class_name

                pred_bboxes.append({
                    'top': bbox['top'],
                    'left': bbox['left'],
                    'height': bbox['height'],
                    'width': bbox['width'],
                    **(
                        {
                            'logits': encode_np_array(logits, flatten=True)
                        } if logits is not None else {}
                    ),
                    **(
                        {
                            'confidence': box_confidence
                        } if box_confidence is not None else {}
                    ),
                    **(
                        {
                            'class_name': box_class_name
                        } if box_class_name is not None else {}
                    ),
                    **(
                        {
                            'entropy': entropy
                        } if entropy is not None else {}
                    ),
                    **(
                        {
                            'ratio_of_confidence': ratio_of_confidence
                        } if ratio_of_confidence is not None else {}
                    ),
                    **(
                        {
                            'margin_of_confidence': margin_of_confidence
                        } if margin_of_confidence is not None else {}
                    ),
                    **(
                        {
                            'objectness': bbox['objectness']
                        } if 'objectness' in bbox else {}
                    ),
                    **(
                        {
                            'embeddings': encode_np_array(bbox['embeddings'], flatten=True)
                        } if 'embeddings' in bbox else {}
                    )
                })

    image_embeddings = json_event.pop('non_encoded_embeddings', None)
    image_height = json_event.get('image_metadata.height', None)
    image_width = json_event.get('image_metadata.height', None)
    pred_box_embeddings = None
    gt_box_embeddings = None

    if image_embeddings is not None and image_height and image_width:
        np_image_embeddings = np.array(image_embeddings)
        if len(np_image_embeddings.shape) == 3:
            if len(pred_bboxes) > 0:
                if 'embeddings' not in pred_bboxes[0]:
                    np_pred_box = np.zeros((len(pred_bboxes), 4))
                    for index, pred_bbox in enumerate(pred_bboxes):
                        np_pred_box[index][0] = pred_bbox['top'] / image_height
                        np_pred_box[index][1] = pred_bbox['left'] / image_width
                        np_pred_box[index][2] = pred_bbox['height'] / \
                            image_height
                        np_pred_box[index][3] = pred_bbox['width'] / \
                            image_width

                    pred_box_embeddings = roi_pooling(
                        np_image_embeddings, np_pred_box, 5)
            if len(gt_bboxes) > 0:
                if 'embeddings' not in gt_bboxes[0]:
                    np_gt_box = np.zeros((len(gt_bboxes), 4))
                    for index, gt_bbox in enumerate(gt_bboxes):
                        np_gt_box[index][0] = gt_bbox['top'] / image_height
                        np_gt_box[index][1] = gt_bbox['left'] / image_width
                        np_gt_box[index][2] = gt_bbox['height'] / image_height
                        np_gt_box[index][3] = gt_bbox['width'] / image_width

                    gt_box_embeddings = roi_pooling(
                        np_image_embeddings, np_gt_box, 5)

    for index, bbox in enumerate(gt_bboxes):
        bbox['id'] = index

    for index, bbox in enumerate(pred_bboxes):
        bbox['id'] = index

    gt_no_match = list(range(len(gt_bboxes)))
    pred_no_match = list(range(len(pred_bboxes)))

    json_event['is_bbox_row'] = False

    json_event_copy = copy.deepcopy(json_event)
    json_event_copy.pop('embeddings', None)
    json_event_copy.pop('non_encoded_embeddings', None)

    if image_embeddings is not None:
        json_event['original_embeddings'] = encode_np_array(image_embeddings)

    matches = [json_event]

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
            copy_event['iou'] = best_iou
            copy_event['is_bbox_row'] = True

            if pred_box_embeddings is not None:
                copy_event['prediction.embeddings'] = encode_np_array(
                    pred_box_embeddings[best_pred['id']], flatten=True)

            if gt_box_embeddings is not None:
                copy_event['groundtruth.embeddings'] = encode_np_array(
                    gt_box_embeddings[gt_bbox['id']], flatten=True)

            matches.append(copy_event)

    for no_match_index in gt_no_match:
        gt_bbox = gt_bboxes[no_match_index]
        copy_event = copy.deepcopy(json_event_copy)
        copy_event['is_bbox_row'] = True
        if gt_box_embeddings is not None:
            copy_event['groundtruth.embeddings'] = encode_np_array(
                gt_box_embeddings[gt_bbox['id']], flatten=True)
        matches.append(copy_event)

    for no_match_index in pred_no_match:
        pred_bbox = pred_bboxes[no_match_index]
        copy_event = copy.deepcopy(json_event_copy)
        copy_event['is_bbox_row'] = True
        if pred_box_embeddings is not None:
            copy_event['prediction.embeddings'] = encode_np_array(
                pred_box_embeddings[pred_bbox['id']], flatten=True)
        matches.append(copy_event)

    return matches


def preprocess_automated_speech_recognition(json_event):

    groundtruth = json_event.pop('groundtruth')
    prediction = json_event.pop('prediction')

    measures = compute_measures(groundtruth['text'], prediction['text'])

    json_event['wer'] = measures['wer']
    json_event['substitutions'] = measures['substitutions']
    json_event['deletions'] = measures['deletions']
    json_event['insertions'] = measures['insertions']
    json_event['hits'] = measures['hits']

    if measures['wer'] == 0:
        json_event['exact_match'] = 1
    else:
        json_event['exact_match'] = 0

    return [json_event]


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


def preprocess_classifier(json_event):
    if not isinstance(json_event['prediction'], dict):  # We can't do anything
        return [json_event]

    prediction = json_event.pop('prediction')

    if 'confidence' in prediction and 'class_name' in prediction:
        max_class_index = compute_argmax(prediction['confidence'])

        json_event['prediction'] = prediction['class_name'][max_class_index]
        json_event['confidence'] = prediction['confidence'][max_class_index]
        json_event['entropy'] = compute_entropy(prediction['confidence'])
        json_event['ratio_of_confidence'] = compute_ratio_of_confidence(
            prediction['confidence'])
        json_event['margin_of_confidence'] = compute_margin_of_confidence(
            prediction['confidence'])

    if 'logits' in prediction:
        json_event['logits'] = encode_np_array(prediction['logits'])

    json_event['prediction_string'] = orjson.dumps(prediction).decode('ascii')

    return [json_event]
