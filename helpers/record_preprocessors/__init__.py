from ..eventprocessor.utils import encode_np_array, decode_to_np_array

from .record_preprocessor import RecordPreprocessor
from .patch_sample import PatchSamplePreprocessor

from schemas.pgsql import models

FeatureVector = models.feature_vector.FeatureVector

def get_preprocessor(config) -> RecordPreprocessor:
    if config is None:
        return None

    type = config.get('type')
    if type is None:
        raise Exception(f"Preprocessor must have a type: {config}")

    if type == 'patch_sample':
        return PatchSamplePreprocessor(**config)
    else:
        raise Exception(f"Unknown preprocessor type: {type}")

def preprocess_to_record(parent_datapoint):
    preprocessor = get_preprocessor(parent_datapoint._preprocessor)
    parent_datapoint._preprocessor = None # don't do it recursively...

    if preprocessor:
        return preprocessor.preprocess_to_record(parent_datapoint)
    else:
        return []
