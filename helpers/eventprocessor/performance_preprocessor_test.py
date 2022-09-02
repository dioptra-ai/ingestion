import os
import orjson
import pytest
from pytest import approx

from performance_preprocessor import (
    preprocess_automated_speech_recognition,
    preprocess_question_answering,
    preprocess_auto_completion,
    preprocess_semantic_similarity,
    preprocess_classifier,
    preprocess_object_detection
)

from utils import (
    compute_iou,
    compute_cosine_similarity,
    encode_np_array,
    decode_np_array
)

@pytest.mark.parametrize(
    'bbox_1, bbox_2, expected_iou',
    [
        [
            {'left': 0, 'top': 10, 'width': 100, 'height': 100},
            {'left': 0, 'top': 10, 'width': 100, 'height': 100},
            1
        ],
        [
            {'left': 0, 'top': 10, 'width': 100, 'height': 100},
            {'left': 200, 'top': 10, 'width': 100, 'height': 100},
            0
        ],
        [
            {'left': 0, 'top': 10, 'width': 100, 'height': 100},
            {'left': 0, 'top': 10, 'width': 50, 'height': 100},
            0.5
        ],
        [
            {'left': 0, 'top': 10, 'width': 100, 'height': 100},
            {'left': 0, 'top': 10, 'width': 50, 'height': 50},
            0.25
        ],
        [
            {'left': 0, 'top': 10, 'width': 100, 'height': 100},
            {'left': 0, 'top': 50, 'width': 50, 'height': 100},
            0.25
        ]
    ]
)
def test_compute_iou(bbox_1, bbox_2, expected_iou):
    result = compute_iou(bbox_1, bbox_2)
    assert result == expected_iou

@pytest.mark.parametrize(
    'list1, list2, expected_cosine_similarity',
    [
        [
            [1, 2, 3],
            [1, 2, 3],
            1
        ],
        [
            [1, 0, 0],
            [0, 0, 1],
            0
        ],
        [
            [1, 10, 100],
            [1, -5, 45],
            0.978
        ]
    ]
)
def test_compute_cosine_similarity(list1, list2, expected_cosine_similarity):
    result = compute_cosine_similarity(list1, list2)
    assert round(result, 3) == expected_cosine_similarity

@pytest.mark.parametrize(
    'json_event, expected_answer',
    [
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'q_n_a',
                'model_version': 'some ver',
                'prediction': [
                    {
                        'text': 's te',
                        'embeddings': [1, 2, 3],
                    },
                    {
                        'text': 'so tex',
                        'embeddings': [1, 5, 6],
                    },
                    {
                        'text': 'som tos',
                        'embeddings': [2, 4, 9],
                    }
                ],
                'groundtruth': {
                    'text': 'some text',
                    'embeddings': [1, 3, 5]
                }
            },
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'q_n_a',
                'model_version': 'some ver',
                'groundtruth.text': 'some text',
                'groundtruth.embeddings': 'BCJNGGhABgAAAAAAAABZBgAAgAA8AEIARQAAAAA=',
                'prediction.text': 's te',
                'prediction.embeddings': 'BCJNGGhABgAAAAAAAABZBgAAgAA8AEAAQgAAAAA=',
                'f1_score': 0,
                'exact_match': 0,
                'semantic_similarity': 0.9938586931957764
            }
        ]
    ]
)
def test_preprocess_question_answering(json_event, expected_answer):
    result = preprocess_question_answering(json_event)
    assert result[0]['request_id'] == expected_answer['request_id']
    assert result[0]['model_id'] == expected_answer['model_id']
    assert result[0]['model_version'] == expected_answer['model_version']
    assert round(result[0]['semantic_similarity'], 3) == round(expected_answer['semantic_similarity'], 3)

@pytest.mark.parametrize(
    'json_event, expected_answer',
    [
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'asr',
                'model_version': 'v1.1',
                'prediction': {'text': 'hello how are you'},
                'groundtruth': {'text': 'hello how are you ?'}
            },
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'asr',
                'model_version': 'v1.1',
                'groundtruth.text': 'hello how are you ?',
                'prediction.text': 'hello how are you',
                'wer': 1/5,
                'substitutions': 0,
                'deletions': 1,
                'insertions': 0,
                'hits': 4,
                'exact_match': 0
            }
        ]
    ]
)
def test_preprocess_automated_speech_recognition(json_event, expected_answer):
    result = preprocess_automated_speech_recognition(json_event)[0]
    assert result == expected_answer


@pytest.mark.parametrize(
    'json_event, expected_answer',
    [
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'auto_complete',
                'model_version': 'v1.1',
                'text': 'hello how are you',
                'prediction': [{'text': 'I\'m fine'}],
                'groundtruth': {'text': '? I\'m fine and you ?'}
            },
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'auto_complete',
                'model_version': 'v1.1',
                'text': 'hello how are you',
                'groundtruth.text': '? I\'m fine and you ?',
                'prediction.text': 'I\'m fine',
                'f1_score': 0.8571428571428571,
                'exact_match': 0.0,
                'best_match_token_length': 4
            }
        ],
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'auto_complete',
                'model_version': 'v1.1',
                'text': 'From which countries did the Norse originate?',
                'prediction': [{'text': 'Denmark, Iceland and Norway', 'displayed': True}],
                'groundtruth': {
                    'text': ' Denmark, Iceland and Norway'}
            },
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'auto_complete',
                'model_version': 'v1.1',
                'text': 'From which countries did the Norse originate?',
                'groundtruth.text': ' Denmark, Iceland and Norway',
                'prediction.text': 'Denmark, Iceland and Norway',
                'f1_score': 1.0,
                'exact_match': 1.0,
                'best_match_token_length': 5
            }

        ]
    ]
)
def test_preprocess_auto_completion(json_event, expected_answer):
    result = preprocess_auto_completion(json_event)[0]
    assert result == expected_answer

@pytest.mark.parametrize(
    'json_event, expected_answer',
    [
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'semantic_similarity',
                'model_version': 'v1.1',
                'prediction': 0.8,
                'groundtruth': 0.9,
                'input_data': {
                    'text_1': 'hello',
                    'embeddings_1': [1, 3, 4],
                    'text_2': 'hello',
                    'embeddings_2': [1, 3, 5]
                }
            },
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'semantic_similarity',
                'model_version': 'v1.1',
                'groundtruth': 0.9,
                'prediction': 0.8,
                'cosine_score': 0.9944903161976939,
                'embeddings': encode_np_array([1, 3, 4]),
                'input_data.text_1': 'hello',
                'input_data.embeddings_1': encode_np_array([1, 3, 4]),
                'input_data.text_2': 'hello',
                'input_data.embeddings_2': encode_np_array([1, 3, 5])
            }
        ]
    ]
)
def test_preprocess_semantic_similarity(json_event, expected_answer):
    result = preprocess_semantic_similarity(json_event)[0]
    assert result == expected_answer


@pytest.mark.parametrize(
    'json_event, expected_answer',
    [
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'classifier',
                'model_version': 'v1.1',
                'prediction': 'my_class_1',
                'groundtruth': 'my_class_2',
                'confidence': 0.1
            },
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'classifier',
                'model_version': 'v1.1',
                'groundtruth': 'my_class_2',
                'prediction': 'my_class_1',
                'confidence': 0.1
            }
        ],
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'classifier',
                'model_version': 'v1.1',
                'prediction': {
                    'class_name': ['my_class_1', 'my_class_2', 'my_class_3'],
                    'confidence': [0.9966208234093001, 0.002470376035336821, 0.0009088005553630329],
                    'logits': [8, 2, 1]
                },
                'groundtruth': 'my_class_2',
                'confidence': 0.1
            },
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'classifier',
                'model_version': 'v1.1',
                'groundtruth': 'my_class_2',
                'prediction': 'my_class_1',
                'confidence': 0.9966208234093001,
                'entropy': 0.022363448193388643,
                'logits': encode_np_array([8, 2, 1]),
                'ratio_of_confidence': 403.42879349273517,
                'margin_of_confidence': 0.0058495526260367026,
                'prediction_string': orjson.dumps({
                    'class_name': ['my_class_1', 'my_class_2', 'my_class_3'],
                    'confidence': [0.9966208234093001, 0.002470376035336821, 0.0009088005553630329],
                    'logits': [8, 2, 1]
                }).decode('ascii')
            }
        ]
    ]
)
def test_preprocess_classifier(json_event, expected_answer):
    result = preprocess_classifier(json_event)[0]
    assert result == expected_answer


@pytest.mark.parametrize(
    'json_event, expected_answer',
    [
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'object_detection',
                'model_type': 'OBJECT_DETECTION',
                'model_version': 'v1.1',
                'prediction': [
                    {
                        'top': 10,
                        'left': 10,
                        'width': 10,
                        'height': 10,
                        'class_name': ['class_1', 'class_2'],
                        'logits': [1.5, 0.1],
                        'objectness': 0.8
                    },
                    {
                        'top': 20,
                        'left': 20,
                        'width': 100,
                        'height': 100,
                        'class_name': ['class_1', 'class_2'],
                        'logits': [0.1, 0.5],
                        'objectness': 0.6
                    }
                ],
                'groundtruth': [
                    {
                        'top': 10,
                        'left': 10,
                        'width': 100,
                        'height': 100,
                        'class_name': 'class_1'
                    }
                ],
                'non_encoded_embeddings': [
                    [[1, 2, 3], [2, 2, 3], [3, 2, 3], [5, 2, 3], [6, 2, 3], [7, 2, 3], [8, 2, 3], [9, 2, 3]],
                    [[11, 2, 3], [12, 2, 3], [13, 2, 3], [15, 2, 3], [16, 2, 3], [17, 2, 3], [18, 2, 3], [19, 2, 3]],
                    [[21, 2, 3], [22, 2, 3], [23, 2, 3], [25, 2, 3], [26, 2, 3], [27, 2, 3], [28, 2, 3], [29, 2, 3]],
                    [[31, 2, 3], [32, 2, 3], [33, 2, 3], [35, 2, 3], [36, 2, 3], [37, 2, 3], [38, 2, 3], [39, 2, 3]],
                    [[41, 2, 3], [42, 2, 3], [43, 2, 3], [45, 2, 3], [46, 2, 3], [47, 2, 3], [48, 2, 3], [49, 2, 3]],
                    [[51, 2, 3], [52, 2, 3], [53, 2, 3], [55, 2, 3], [56, 2, 3], [57, 2, 3], [58, 2, 3], [59, 2, 3]],
                    [[61, 2, 3], [62, 2, 3], [63, 2, 3], [65, 2, 3], [66, 2, 3], [67, 2, 3], [68, 2, 3], [69, 2, 3]],
                    [[71, 2, 3], [72, 2, 3], [73, 2, 3], [75, 2, 3], [76, 2, 3], [77, 2, 3], [78, 2, 3], [79, 2, 3]]
                ],
                'image_metadata.height': 200,
                'image_metadata.wight': 300
            },
            [
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': False, 'original_embeddings': 'BCJNGGhAAAIAAAAAAAC9VQEAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/w5GYWxzZSwgJ3NoYXBlJzogKDgsIDgsIDMpLCB9IAEAI5EKADwAQABCAEAGABFCBgARRQYAEUYGABFHBgAQSAYAEoAGABFJEgARSgwAAgYAEUsSABBMBgAhQEwSAAEGABLAEgARTRIAEU0SABFNEgARThIAEU4SABFOPAARTxgAEU8SABFPEgAQUAYAESAGACFgUDAAAQYAIaBQJAABBgAS4CQAEVE8ABFRKgARUSQAEVEkABFRJAARUU4AEVIqABFSJAARUk4AEVIqABFSJAARUiQAEVMkABFTTgARUyoAEVMkABFTTgARUyoAAQYAIRBUKgABBgAhMFQwAAEGABFQBgAhcFRgAAEGABGQBgAhsFRCAAEGACHQVEgAsFQAQABC8FQAQABCAAAAAA=='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'iou': 0.680672268907563, 'prediction.top': 20, 'prediction.left': 20, 'prediction.height': 100, 'prediction.width': 100, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCmYuADgAAAAA', 'prediction.confidence': 0.598687660112452, 'prediction.class_name': 'class_2', 'prediction.entropy': 0.9717130895791362, 'prediction.ratio_of_confidence': 1.4918246976412703, 'prediction.margin_of_confidence': 0.802624679775096, 'prediction.objectness': 0.6, 'prediction.id': 1, 'groundtruth.top': 10, 'groundtruth.left': 10, 'groundtruth.height': 100, 'groundtruth.width': 100, 'groundtruth.class_name': 'class_1', 'groundtruth.id': 0, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbugAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAngQoASgBAAEJaBgAgAEsGACGmSwwAEEwGABHABgAR7QYAIEBNBgAhk00YAAIGABFOHgARTh4AEU8eABFPGAABBgAgYFAGABF2BgARoAYAEcoGACHgUDAAEFEGABFWBgARgAYAwKpRAEAAQsBRAEAAQgAAAAA=', 'groundtruth.embeddings': 'BCJNGGhAFgEAAAAAAADbugAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAngQoASgBAAEJaBgAgAEsGACGmSwwAEEwGABHABgAR7QYAIEBNBgAhk00YAAIGABFOHgARTh4AEU8eABFPGAABBgAgYFAGABF2BgARoAYAEcoGACHgUDAAEFEGABFWBgARgAYAwKpRAEAAQsBRAEAAQgAAAAA='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 10, 'prediction.left': 10, 'prediction.height': 10, 'prediction.width': 10, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCgA+Zi4AAAAA', 'prediction.confidence': 0.8021838885585817, 'prediction.class_name': 'class_1', 'prediction.entropy': 0.7175387563932007, 'prediction.ratio_of_confidence': 4.055199966844674, 'prediction.margin_of_confidence': 0.3956322228828366, 'prediction.objectness': 0.8, 'prediction.id': 0, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbWwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAnfwoASgBAAEIGAHhQSgBAAEIAAAAA'}
            ]
        ],
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'object_detection',
                'model_type': 'OBJECT_DETECTION',
                'model_version': 'v1.1',
                'prediction': [
                    {
                        'top': 10,
                        'left': 10,
                        'width': 10,
                        'height': 10,
                        'class_name': ['class_1', 'class_2'],
                        'logits': [1.5, 0.1],
                        'objectness': 0.8
                    },
                    {
                        'top': 20,
                        'left': 20,
                        'width': 100,
                        'height': 100,
                        'class_name': ['class_1', 'class_2'],
                        'logits': [0.1, 0.5],
                        'objectness': 0.6
                    }
                ],
                'non_encoded_embeddings': [
                    [[1, 2, 3], [2, 2, 3], [3, 2, 3], [5, 2, 3], [6, 2, 3], [7, 2, 3], [8, 2, 3], [9, 2, 3]],
                    [[11, 2, 3], [12, 2, 3], [13, 2, 3], [15, 2, 3], [16, 2, 3], [17, 2, 3], [18, 2, 3], [19, 2, 3]],
                    [[21, 2, 3], [22, 2, 3], [23, 2, 3], [25, 2, 3], [26, 2, 3], [27, 2, 3], [28, 2, 3], [29, 2, 3]],
                    [[31, 2, 3], [32, 2, 3], [33, 2, 3], [35, 2, 3], [36, 2, 3], [37, 2, 3], [38, 2, 3], [39, 2, 3]],
                    [[41, 2, 3], [42, 2, 3], [43, 2, 3], [45, 2, 3], [46, 2, 3], [47, 2, 3], [48, 2, 3], [49, 2, 3]],
                    [[51, 2, 3], [52, 2, 3], [53, 2, 3], [55, 2, 3], [56, 2, 3], [57, 2, 3], [58, 2, 3], [59, 2, 3]],
                    [[61, 2, 3], [62, 2, 3], [63, 2, 3], [65, 2, 3], [66, 2, 3], [67, 2, 3], [68, 2, 3], [69, 2, 3]],
                    [[71, 2, 3], [72, 2, 3], [73, 2, 3], [75, 2, 3], [76, 2, 3], [77, 2, 3], [78, 2, 3], [79, 2, 3]]
                ],
                'image_metadata.height': 200,
                'image_metadata.wight': 300
            },
            [
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': False, 'original_embeddings': 'BCJNGGhAAAIAAAAAAAC9VQEAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/w5GYWxzZSwgJ3NoYXBlJzogKDgsIDgsIDMpLCB9IAEAI5EKADwAQABCAEAGABFCBgARRQYAEUYGABFHBgAQSAYAEoAGABFJEgARSgwAAgYAEUsSABBMBgAhQEwSAAEGABLAEgARTRIAEU0SABFNEgARThIAEU4SABFOPAARTxgAEU8SABFPEgAQUAYAESAGACFgUDAAAQYAIaBQJAABBgAS4CQAEVE8ABFRKgARUSQAEVEkABFRJAARUU4AEVIqABFSJAARUk4AEVIqABFSJAARUiQAEVMkABFTTgARUyoAEVMkABFTTgARUyoAAQYAIRBUKgABBgAhMFQwAAEGABFQBgAhcFRgAAEGABGQBgAhsFRCAAEGACHQVEgAsFQAQABC8FQAQABCAAAAAA=='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 10, 'prediction.left': 10, 'prediction.height': 10, 'prediction.width': 10, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCgA+Zi4AAAAA', 'prediction.confidence': 0.8021838885585817, 'prediction.class_name': 'class_1', 'prediction.entropy': 0.7175387563932007, 'prediction.ratio_of_confidence': 4.055199966844674, 'prediction.margin_of_confidence': 0.3956322228828366, 'prediction.objectness': 0.8, 'prediction.id': 0, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbWwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAnfwoASgBAAEIGAHhQSgBAAEIAAAAA'},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 20, 'prediction.left': 20, 'prediction.height': 100, 'prediction.width': 100, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCmYuADgAAAAA', 'prediction.confidence': 0.598687660112452, 'prediction.class_name': 'class_2', 'prediction.entropy': 0.9717130895791362, 'prediction.ratio_of_confidence': 1.4918246976412703, 'prediction.margin_of_confidence': 0.802624679775096, 'prediction.objectness': 0.6, 'prediction.id': 1, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbugAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAngQoASgBAAEJaBgAgAEsGACGmSwwAEEwGABHABgAR7QYAIEBNBgAhk00YAAIGABFOHgARTh4AEU8eABFPGAABBgAgYFAGABF2BgARoAYAEcoGACHgUDAAEFEGABFWBgARgAYAwKpRAEAAQsBRAEAAQgAAAAA='}
            ]
        ],
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'object_detection',
                'model_type': 'OBJECT_DETECTION',
                'model_version': 'v1.1',
                'prediction': {
                    'boxes': [[10, 10, 10, 10], [20, 20, 100, 100]],
                    'logits': [[1.5, 0.1], [0.1, 0.5]],
                    'class_names': ['class_1', 'class_2'],
                    'objectness': [0.8, 0.6]
                },
                'groundtruth': {
                    'boxes': [[10, 10, 100, 100]],
                    'class_names': ['class_1']
                },
                'non_encoded_embeddings': [
                    [[1, 2, 3], [2, 2, 3], [3, 2, 3], [5, 2, 3], [6, 2, 3], [7, 2, 3], [8, 2, 3], [9, 2, 3]],
                    [[11, 2, 3], [12, 2, 3], [13, 2, 3], [15, 2, 3], [16, 2, 3], [17, 2, 3], [18, 2, 3], [19, 2, 3]],
                    [[21, 2, 3], [22, 2, 3], [23, 2, 3], [25, 2, 3], [26, 2, 3], [27, 2, 3], [28, 2, 3], [29, 2, 3]],
                    [[31, 2, 3], [32, 2, 3], [33, 2, 3], [35, 2, 3], [36, 2, 3], [37, 2, 3], [38, 2, 3], [39, 2, 3]],
                    [[41, 2, 3], [42, 2, 3], [43, 2, 3], [45, 2, 3], [46, 2, 3], [47, 2, 3], [48, 2, 3], [49, 2, 3]],
                    [[51, 2, 3], [52, 2, 3], [53, 2, 3], [55, 2, 3], [56, 2, 3], [57, 2, 3], [58, 2, 3], [59, 2, 3]],
                    [[61, 2, 3], [62, 2, 3], [63, 2, 3], [65, 2, 3], [66, 2, 3], [67, 2, 3], [68, 2, 3], [69, 2, 3]],
                    [[71, 2, 3], [72, 2, 3], [73, 2, 3], [75, 2, 3], [76, 2, 3], [77, 2, 3], [78, 2, 3], [79, 2, 3]]
                ],
                'image_metadata.height': 200,
                'image_metadata.wight': 300
            },
            [
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': False, 'original_embeddings': 'BCJNGGhAAAIAAAAAAAC9VQEAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/w5GYWxzZSwgJ3NoYXBlJzogKDgsIDgsIDMpLCB9IAEAI5EKADwAQABCAEAGABFCBgARRQYAEUYGABFHBgAQSAYAEoAGABFJEgARSgwAAgYAEUsSABBMBgAhQEwSAAEGABLAEgARTRIAEU0SABFNEgARThIAEU4SABFOPAARTxgAEU8SABFPEgAQUAYAESAGACFgUDAAAQYAIaBQJAABBgAS4CQAEVE8ABFRKgARUSQAEVEkABFRJAARUU4AEVIqABFSJAARUk4AEVIqABFSJAARUiQAEVMkABFTTgARUyoAEVMkABFTTgARUyoAAQYAIRBUKgABBgAhMFQwAAEGABFQBgAhcFRgAAEGABGQBgAhsFRCAAEGACHQVEgAsFQAQABC8FQAQABCAAAAAA=='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'iou': 0.680672268907563, 'prediction.top': 20, 'prediction.left': 20, 'prediction.height': 100, 'prediction.width': 100, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCmYuADgAAAAA', 'prediction.confidence': 0.598687660112452, 'prediction.class_name': 'class_2', 'prediction.entropy': 0.9717130895791362, 'prediction.ratio_of_confidence': 1.4918246976412703, 'prediction.margin_of_confidence': 0.802624679775096, 'prediction.objectness': 0.6, 'prediction.id': 1, 'groundtruth.top': 10, 'groundtruth.left': 10, 'groundtruth.height': 100, 'groundtruth.width': 100, 'groundtruth.class_name': 'class_1', 'groundtruth.id': 0, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbugAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAngQoASgBAAEJaBgAgAEsGACGmSwwAEEwGABHABgAR7QYAIEBNBgAhk00YAAIGABFOHgARTh4AEU8eABFPGAABBgAgYFAGABF2BgARoAYAEcoGACHgUDAAEFEGABFWBgARgAYAwKpRAEAAQsBRAEAAQgAAAAA=', 'groundtruth.embeddings': 'BCJNGGhAFgEAAAAAAADbugAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAngQoASgBAAEJaBgAgAEsGACGmSwwAEEwGABHABgAR7QYAIEBNBgAhk00YAAIGABFOHgARTh4AEU8eABFPGAABBgAgYFAGABF2BgARoAYAEcoGACHgUDAAEFEGABFWBgARgAYAwKpRAEAAQsBRAEAAQgAAAAA='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 10, 'prediction.left': 10, 'prediction.height': 10, 'prediction.width': 10, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCgA+Zi4AAAAA', 'prediction.confidence': 0.8021838885585817, 'prediction.class_name': 'class_1', 'prediction.entropy': 0.7175387563932007, 'prediction.ratio_of_confidence': 4.055199966844674, 'prediction.margin_of_confidence': 0.3956322228828366, 'prediction.objectness': 0.8, 'prediction.id': 0, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbWwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAnfwoASgBAAEIGAHhQSgBAAEIAAAAA'}
            ]
        ],
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'object_detection',
                'model_type': 'OBJECT_DETECTION',
                'model_version': 'v1.1',
                'prediction': {
                    'boxes': [[10, 10, 10, 10], [20, 20, 100, 100]],
                    'logits': [[1.5, 0.1], [0.1, 0.5]],
                    'class_names': ['class_1', 'class_2'],
                    'objectness': [0.8, 0.6]
                },
                'non_encoded_embeddings': [
                    [[1, 2, 3], [2, 2, 3], [3, 2, 3], [5, 2, 3], [6, 2, 3], [7, 2, 3], [8, 2, 3], [9, 2, 3]],
                    [[11, 2, 3], [12, 2, 3], [13, 2, 3], [15, 2, 3], [16, 2, 3], [17, 2, 3], [18, 2, 3], [19, 2, 3]],
                    [[21, 2, 3], [22, 2, 3], [23, 2, 3], [25, 2, 3], [26, 2, 3], [27, 2, 3], [28, 2, 3], [29, 2, 3]],
                    [[31, 2, 3], [32, 2, 3], [33, 2, 3], [35, 2, 3], [36, 2, 3], [37, 2, 3], [38, 2, 3], [39, 2, 3]],
                    [[41, 2, 3], [42, 2, 3], [43, 2, 3], [45, 2, 3], [46, 2, 3], [47, 2, 3], [48, 2, 3], [49, 2, 3]],
                    [[51, 2, 3], [52, 2, 3], [53, 2, 3], [55, 2, 3], [56, 2, 3], [57, 2, 3], [58, 2, 3], [59, 2, 3]],
                    [[61, 2, 3], [62, 2, 3], [63, 2, 3], [65, 2, 3], [66, 2, 3], [67, 2, 3], [68, 2, 3], [69, 2, 3]],
                    [[71, 2, 3], [72, 2, 3], [73, 2, 3], [75, 2, 3], [76, 2, 3], [77, 2, 3], [78, 2, 3], [79, 2, 3]]
                ],
                'image_metadata.height': 200,
                'image_metadata.wight': 300
            },
            [
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': False, 'original_embeddings': 'BCJNGGhAAAIAAAAAAAC9VQEAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/w5GYWxzZSwgJ3NoYXBlJzogKDgsIDgsIDMpLCB9IAEAI5EKADwAQABCAEAGABFCBgARRQYAEUYGABFHBgAQSAYAEoAGABFJEgARSgwAAgYAEUsSABBMBgAhQEwSAAEGABLAEgARTRIAEU0SABFNEgARThIAEU4SABFOPAARTxgAEU8SABFPEgAQUAYAESAGACFgUDAAAQYAIaBQJAABBgAS4CQAEVE8ABFRKgARUSQAEVEkABFRJAARUU4AEVIqABFSJAARUk4AEVIqABFSJAARUiQAEVMkABFTTgARUyoAEVMkABFTTgARUyoAAQYAIRBUKgABBgAhMFQwAAEGABFQBgAhcFRgAAEGABGQBgAhsFRCAAEGACHQVEgAsFQAQABC8FQAQABCAAAAAA=='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 10, 'prediction.left': 10, 'prediction.height': 10, 'prediction.width': 10, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCgA+Zi4AAAAA', 'prediction.confidence': 0.8021838885585817, 'prediction.class_name': 'class_1', 'prediction.entropy': 0.7175387563932007, 'prediction.ratio_of_confidence': 4.055199966844674, 'prediction.margin_of_confidence': 0.3956322228828366, 'prediction.objectness': 0.8, 'prediction.id': 0, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbWwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAnfwoASgBAAEIGAHhQSgBAAEIAAAAA'},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 20, 'prediction.left': 20, 'prediction.height': 100, 'prediction.width': 100, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCmYuADgAAAAA', 'prediction.confidence': 0.598687660112452, 'prediction.class_name': 'class_2', 'prediction.entropy': 0.9717130895791362, 'prediction.ratio_of_confidence': 1.4918246976412703, 'prediction.margin_of_confidence': 0.802624679775096, 'prediction.objectness': 0.6, 'prediction.id': 1, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbugAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAngQoASgBAAEJaBgAgAEsGACGmSwwAEEwGABHABgAR7QYAIEBNBgAhk00YAAIGABFOHgARTh4AEU8eABFPGAABBgAgYFAGABF2BgARoAYAEcoGACHgUDAAEFEGABFWBgARgAYAwKpRAEAAQsBRAEAAQgAAAAA='}
            ]
        ],
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'object_detection',
                'model_type': 'OBJECT_DETECTION',
                'model_version': 'v1.1',
                'prediction': {
                    'boxes': [[10, 10, 10, 10], [20, 20, 100, 100]],
                    'logits': [[1.5, 0.1], [0.1, 0.5]],
                    'class_names': ['class_1', 'class_2'],
                    'objectness': [0.8, 0.6]
                },
                'groundtruth': {
                    'boxes': [[10, 10, 100, 100]],
                    'class_names': ['class_1']
                },
                'non_encoded_embeddings': [
                    [[1, 2, 3], [2, 2, 3], [3, 2, 3], [5, 2, 3], [6, 2, 3], [7, 2, 3], [8, 2, 3], [9, 2, 3]],
                    [[11, 2, 3], [12, 2, 3], [13, 2, 3], [15, 2, 3], [16, 2, 3], [17, 2, 3], [18, 2, 3], [19, 2, 3]],
                    [[21, 2, 3], [22, 2, 3], [23, 2, 3], [25, 2, 3], [26, 2, 3], [27, 2, 3], [28, 2, 3], [29, 2, 3]],
                    [[31, 2, 3], [32, 2, 3], [33, 2, 3], [35, 2, 3], [36, 2, 3], [37, 2, 3], [38, 2, 3], [39, 2, 3]],
                    [[41, 2, 3], [42, 2, 3], [43, 2, 3], [45, 2, 3], [46, 2, 3], [47, 2, 3], [48, 2, 3], [49, 2, 3]],
                    [[51, 2, 3], [52, 2, 3], [53, 2, 3], [55, 2, 3], [56, 2, 3], [57, 2, 3], [58, 2, 3], [59, 2, 3]],
                    [[61, 2, 3], [62, 2, 3], [63, 2, 3], [65, 2, 3], [66, 2, 3], [67, 2, 3], [68, 2, 3], [69, 2, 3]],
                    [[71, 2, 3], [72, 2, 3], [73, 2, 3], [75, 2, 3], [76, 2, 3], [77, 2, 3], [78, 2, 3], [79, 2, 3]]
                ],
                'image_metadata.height': 200,
                'image_metadata.wight': 300
            },
            [
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': False, 'original_embeddings': 'BCJNGGhAAAIAAAAAAAC9VQEAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/w5GYWxzZSwgJ3NoYXBlJzogKDgsIDgsIDMpLCB9IAEAI5EKADwAQABCAEAGABFCBgARRQYAEUYGABFHBgAQSAYAEoAGABFJEgARSgwAAgYAEUsSABBMBgAhQEwSAAEGABLAEgARTRIAEU0SABFNEgARThIAEU4SABFOPAARTxgAEU8SABFPEgAQUAYAESAGACFgUDAAAQYAIaBQJAABBgAS4CQAEVE8ABFRKgARUSQAEVEkABFRJAARUU4AEVIqABFSJAARUk4AEVIqABFSJAARUiQAEVMkABFTTgARUyoAEVMkABFTTgARUyoAAQYAIRBUKgABBgAhMFQwAAEGABFQBgAhcFRgAAEGABGQBgAhsFRCAAEGACHQVEgAsFQAQABC8FQAQABCAAAAAA=='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'iou': 0.680672268907563, 'prediction.top': 20, 'prediction.left': 20, 'prediction.height': 100, 'prediction.width': 100, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCmYuADgAAAAA', 'prediction.confidence': 0.598687660112452, 'prediction.class_name': 'class_2', 'prediction.entropy': 0.9717130895791362, 'prediction.ratio_of_confidence': 1.4918246976412703, 'prediction.margin_of_confidence': 0.802624679775096, 'prediction.objectness': 0.6, 'prediction.id': 1, 'groundtruth.top': 10, 'groundtruth.left': 10, 'groundtruth.height': 100, 'groundtruth.width': 100, 'groundtruth.class_name': 'class_1', 'groundtruth.id': 0, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbugAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAngQoASgBAAEJaBgAgAEsGACGmSwwAEEwGABHABgAR7QYAIEBNBgAhk00YAAIGABFOHgARTh4AEU8eABFPGAABBgAgYFAGABF2BgARoAYAEcoGACHgUDAAEFEGABFWBgARgAYAwKpRAEAAQsBRAEAAQgAAAAA=', 'groundtruth.embeddings': 'BCJNGGhAFgEAAAAAAADbugAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAngQoASgBAAEJaBgAgAEsGACGmSwwAEEwGABHABgAR7QYAIEBNBgAhk00YAAIGABFOHgARTh4AEU8eABFPGAABBgAgYFAGABF2BgARoAYAEcoGACHgUDAAEFEGABFWBgARgAYAwKpRAEAAQsBRAEAAQgAAAAA='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 10, 'prediction.left': 10, 'prediction.height': 10, 'prediction.width': 10, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCgA+Zi4AAAAA', 'prediction.confidence': 0.8021838885585817, 'prediction.class_name': 'class_1', 'prediction.entropy': 0.7175387563932007, 'prediction.ratio_of_confidence': 4.055199966844674, 'prediction.margin_of_confidence': 0.3956322228828366, 'prediction.objectness': 0.8, 'prediction.id': 0, 'prediction.embeddings': 'BCJNGGhAFgEAAAAAAADbWwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wpGYWxzZSwgJ3NoYXBlJzogKDc1LCksIH0gAQAnfwoASgBAAEIGAHhQSgBAAEIAAAAA'}
            ]
        ],
        [
            {
                'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1',
                'model_id': 'object_detection',
                'model_type': 'OBJECT_DETECTION',
                'model_version': 'v1.1',
                'prediction': {
                    'boxes': [[10, 10, 10, 10], [20, 20, 100, 100]],
                    'logits': [[1.5, 0.1], [0.1, 0.5]],
                    'class_names': ['class_1', 'class_2'],
                    'objectness': [0.8, 0.6],
                    'embeddings': [[0, 1], [0, 1]]
                },
                'non_encoded_embeddings': [
                    [[1, 2, 3], [2, 2, 3], [3, 2, 3], [5, 2, 3], [6, 2, 3], [7, 2, 3], [8, 2, 3], [9, 2, 3]],
                    [[11, 2, 3], [12, 2, 3], [13, 2, 3], [15, 2, 3], [16, 2, 3], [17, 2, 3], [18, 2, 3], [19, 2, 3]],
                    [[21, 2, 3], [22, 2, 3], [23, 2, 3], [25, 2, 3], [26, 2, 3], [27, 2, 3], [28, 2, 3], [29, 2, 3]],
                    [[31, 2, 3], [32, 2, 3], [33, 2, 3], [35, 2, 3], [36, 2, 3], [37, 2, 3], [38, 2, 3], [39, 2, 3]],
                    [[41, 2, 3], [42, 2, 3], [43, 2, 3], [45, 2, 3], [46, 2, 3], [47, 2, 3], [48, 2, 3], [49, 2, 3]],
                    [[51, 2, 3], [52, 2, 3], [53, 2, 3], [55, 2, 3], [56, 2, 3], [57, 2, 3], [58, 2, 3], [59, 2, 3]],
                    [[61, 2, 3], [62, 2, 3], [63, 2, 3], [65, 2, 3], [66, 2, 3], [67, 2, 3], [68, 2, 3], [69, 2, 3]],
                    [[71, 2, 3], [72, 2, 3], [73, 2, 3], [75, 2, 3], [76, 2, 3], [77, 2, 3], [78, 2, 3], [79, 2, 3]]
                ],
                'image_metadata.height': 200,
                'image_metadata.wight': 300
            },
            [
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': False, 'original_embeddings': 'BCJNGGhAAAIAAAAAAAC9VQEAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/w5GYWxzZSwgJ3NoYXBlJzogKDgsIDgsIDMpLCB9IAEAI5EKADwAQABCAEAGABFCBgARRQYAEUYGABFHBgAQSAYAEoAGABFJEgARSgwAAgYAEUsSABBMBgAhQEwSAAEGABLAEgARTRIAEU0SABFNEgARThIAEU4SABFOPAARTxgAEU8SABFPEgAQUAYAESAGACFgUDAAAQYAIaBQJAABBgAS4CQAEVE8ABFRKgARUSQAEVEkABFRJAARUU4AEVIqABFSJAARUk4AEVIqABFSJAARUiQAEVMkABFTTgARUyoAEVMkABFTTgARUyoAAQYAIRBUKgABBgAhMFQwAAEGABFQBgAhcFRgAAEGABGQBgAhsFRCAAEGACHQVEgAsFQAQABC8FQAQABCAAAAAA=='},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 10, 'prediction.left': 10, 'prediction.height': 10, 'prediction.width': 10, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCgA+Zi4AAAAA', 'prediction.confidence': 0.8021838885585817, 'prediction.class_name': 'class_1', 'prediction.entropy': 0.7175387563932007, 'prediction.ratio_of_confidence': 4.055199966844674, 'prediction.margin_of_confidence': 0.3956322228828366, 'prediction.objectness': 0.8, 'prediction.id': 0, 'prediction.embeddings': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCgAAADwAAAAA'},
                {'request_id': '603e6a02-cfc7-4b1d-bdf0-64ab0adf41e1', 'model_id': 'object_detection', 'model_type': 'OBJECT_DETECTION', 'model_version': 'v1.1', 'image_metadata.height': 200, 'image_metadata.wight': 300, 'is_bbox_row': True, 'prediction.top': 20, 'prediction.left': 20, 'prediction.height': 100, 'prediction.width': 100, 'prediction.logits': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCmYuADgAAAAA', 'prediction.confidence': 0.598687660112452, 'prediction.class_name': 'class_2', 'prediction.entropy': 0.9717130895791362, 'prediction.ratio_of_confidence': 1.4918246976412703, 'prediction.margin_of_confidence': 0.802624679775096, 'prediction.objectness': 0.6, 'prediction.id': 1, 'prediction.embeddings': 'BCJNGGhAhAAAAAAAAACrTwAAAPAZk05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZRgA/wlGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSABAChQCgAAADwAAAAA'}
            ]
        ]
    ]
)
def test_preprocess_object_detection(json_event, expected_answer):
    results = preprocess_object_detection(json_event)
    assert decode_np_array(results[0]['original_embeddings']).shape == (8, 8, 3)

    for index, result in enumerate(results):
        for field in ['confidence', 'ratio_of_confidence', 'entropy', 'margin_of_confidence']:
            my_field = 'prediction.' + field
            if my_field in result:
                assert result[my_field] == approx(expected_answer[index][my_field])
                result.pop(my_field)
                expected_answer[index].pop(my_field)

    assert results == expected_answer
