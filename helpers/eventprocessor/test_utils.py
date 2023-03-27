import pytest
import json
import os

from helpers.eventprocessor.utils import compute_mean, encode_np_array

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'test_data')

from helpers.eventprocessor.utils import process_logits

def test_process_logits_binary_classifier():
    logits = [0.1]
    class_names = ['positive', 'negative']
    results = process_logits(logits, class_names)
    expected_results = {
        'confidences': pytest.approx([0.52497918747894, 0.47502081252106]),
        'confidence': pytest.approx(0.52497918747894),
        'class_name': 'positive',
        'entropy': pytest.approx(0.9981988829078698)
    }

    assert results == expected_results


def test_process_logits_multi_class_classifier():
    logits = [0.1, 0.2, 0.8]
    class_names = ['positive', 'negative', 'neutral']
    results = process_logits(logits, class_names)
    expected_results = {
        'confidences': pytest.approx([0.2427818748077445, 0.2683154674734019, 0.4889026577188538]),
        'confidence': pytest.approx(0.4889026577188538),
        'class_name': 'neutral',
        'entropy': pytest.approx(0.9525912616642588)
    }

    assert results == expected_results

def test_process_logits_segmentation():
    with open(os.path.join(TEST_DATA_DIR, 'semseg_logits.json'), 'r') as file:
        logits = json.load(file)
    with open(os.path.join(TEST_DATA_DIR, 'semseg_confidences.json'), 'r') as file:
        confidences = json.load(file)
    with open(os.path.join(TEST_DATA_DIR, 'semseg_entropy.json'), 'r') as file:
        entropy = json.load(file)
    with open(os.path.join(TEST_DATA_DIR, 'semseg_class_mask.json'), 'r') as file:
        class_mask = json.load(file)
    class_names = ['positive', 'negative', 'neutral']
    results = process_logits(logits, class_names)
    expected_results = {
        'confidences': pytest.approx(confidences),
        'confidence': pytest.approx(compute_mean(confidences)),
        'entropy': pytest.approx(entropy),
        'segmentation_class_mask': class_mask
    }

    #ignoring this entry as we already check the entropy, which is the aggregate
    results.pop('pixel_entropy')

    assert results == expected_results


def test_process_logits_mcdo_segmentation():
    with open(os.path.join(TEST_DATA_DIR, 'mcdo_semseg_logits.json'), 'r') as file:
        logits = json.load(file)
    with open(os.path.join(TEST_DATA_DIR, 'mcdo_semseg_confidences.json'), 'r') as file:
        confidences = json.load(file)
    with open(os.path.join(TEST_DATA_DIR, 'mcdo_semseg_entropy.json'), 'r') as file:
        entropy = json.load(file)
    with open(os.path.join(TEST_DATA_DIR, 'mcdo_semseg_variance.json'), 'r') as file:
        variance = json.load(file)
    with open(os.path.join(TEST_DATA_DIR, 'mcdo_semseg_class_mask.json'), 'r') as file:
        class_mask = json.load(file)
    class_names = ['positive', 'negative', 'neutral']
    results = process_logits(logits, class_names)
    expected_results = {
        'confidences': pytest.approx(confidences),
        'confidence': pytest.approx(compute_mean(confidences)),
        'entropy': pytest.approx(entropy),
        'variance': pytest.approx(variance),
        'segmentation_class_mask': class_mask,
    }

    #ignoring these entries as we already check the entropy and variance, which are the aggregates
    results.pop('pixel_entropy')
    results.pop('pixel_variance')

    assert results == expected_results