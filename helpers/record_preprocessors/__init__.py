
from schemas.pgsql import models

FeatureVector = models.feature_vector.FeatureVector

class RecordPreprocessor():
    def __init__(self, config):
        self.config = config
    
    def process_datapoint(self, datapoint, pg_session):
        pass

def _get_preprocessor(config) -> RecordPreprocessor:
    if config is None:
        return None

    type = config.get('type')
    if type is None:
        raise Exception(f"Preprocessor must have a type: {config}")

    if type == 'patch_sample':
        # This is a hack to avoid circular imports (patch_sample imports app which imports this file).
        # Apparently fine: https://stackoverflow.com/questions/12487549/how-safe-is-it-to-import-a-module-multiple-times
        from .patch_sample import PatchSamplePreprocessor

        return PatchSamplePreprocessor(**config)
    else:
        raise Exception(f"Unknown preprocessor type: {type}")

def preprocess_datapoint(datapoint, pg_session):
    preprocessor = _get_preprocessor(datapoint._preprocessor)
    if preprocessor is None:
        return []

    return preprocessor.process_datapoint(datapoint, pg_session)
