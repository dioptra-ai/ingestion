import os
import json

import pytest
from pytest import approx

from event_processor import process_event
from utils import encode_np_array

RESOURCES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'resources')


with open(os.path.join(RESOURCES_DIR, 'test_events.json'), 'r') as file:
    TEST_CASES = json.load(file)

@pytest.mark.parametrize(
    'test_event, expected_results',
    zip(
        TEST_CASES['test_events'],
        TEST_CASES['expected_results']))
def test_process_event(test_event, expected_results):

    if test_event['model_id'] == 'object_detection':
        os.environ['MAX_EMBEDDINGS_SIZE'] = '100'

    if 'embeddings' in test_event:
        test_event['embeddings'] = encode_np_array(test_event['embeddings'])

    results = process_event(json.dumps(test_event))

    for index, result in enumerate(results):
        result.pop('processing_timestamp')
        result.pop('uuid')
        for field in ['confidence', 'ratio_of_confidence', 'entropy', 'margin_of_confidence']:
            my_field = 'prediction.' + field
            if my_field in result:
                assert result[my_field] == approx(expected_results[index][my_field])
                result.pop(my_field)
                expected_results[index].pop(my_field)

    assert results == expected_results
