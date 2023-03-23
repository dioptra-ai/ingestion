import numpy as np
from copy import deepcopy

from schemas.pgsql import models
from ..eventprocessor.utils import encode_np_array, decode_to_np_array

from helpers.record_preprocessors.record_preprocessor import RecordPreprocessor

Prediction = models.prediction.Prediction
FeatureVector = models.feature_vector.FeatureVector

class PatchSamplePreprocessor(RecordPreprocessor):
    def __init__(self, size, **kwargs):
        self.size = size

    def _get_record_for_patch(self, p_or_gt, patch_x, num_patches_x, patch_y, num_patches_y, pg_session):
        # Hopefully this is cached locally in the session and fast...
        row = pg_session.query(models.datapoint.Datapoint.metadata_).filter(models.datapoint.Datapoint.id == p_or_gt['datapoint']).first()
        metadata = row.metadata_
        height = metadata.get('height', None)
        width = metadata.get('width', None)
        if height is None or width is None:
            raise Exception(f"metadata must contain height and width for patch sampling: {p_or_gt['metadata']}")

        record = deepcopy(p_or_gt)
        del record['id']

        record['top'] = int(patch_y * height / num_patches_y)
        record['left'] = int(patch_x * width / num_patches_x)
        record['height'] = int(height / num_patches_y)
        record['width'] = int(width / num_patches_x)

        if p_or_gt.get('segmentation_class_mask', None) is not None:
            segmentation_class_mask = np.array(p_or_gt.segmentation_class_mask)
            class_mask_shape = segmentation_class_mask.shape
            if len(class_mask_shape) != 2:
                raise Exception("Segmentation class mask must be a 2D array")
            patch_left_in_mask = int(patch_x * class_mask_shape[1] / num_patches_x)
            patch_top_in_mask = int(patch_y * class_mask_shape[0] / num_patches_y)
            patch_width_in_mask = int(class_mask_shape[1] / num_patches_x)
            patch_height_in_mask = int(class_mask_shape[0] / num_patches_y)
            patch_segmentation_class_mask = segmentation_class_mask[patch_left_in_mask:patch_left_in_mask + patch_width_in_mask, patch_top_in_mask:patch_top_in_mask + patch_height_in_mask]
            record['segmentation_class_mask'] = patch_segmentation_class_mask.tolist()

        return record

    def process_prediction(self, record, pg_session):
        num_patches_x = self.size[0]
        num_patches_y = self.size[1]
        new_prediction_records = []
        task_type = record['task_type']

        if task_type == 'SEGMENTATION':
            for patch_x in range(0, num_patches_x):
                for patch_y in range(0, num_patches_y):
                    new_prediction_record = self._get_record_for_patch(record, patch_x, num_patches_x, patch_y, num_patches_y, pg_session=pg_session)

                    # Logits
                    feature_vector = pg_session.query(FeatureVector).filter(
                        FeatureVector.prediction == record['id'],
                        FeatureVector.type == 'LOGITS',
                        FeatureVector.model_name == record['model_name']
                    ).first()
                    if feature_vector is not None:
                        logits = decode_to_np_array(feature_vector.encoded_value)
                        logits_shape = logits.shape
                        if len(logits_shape) != 3:
                            raise Exception("Logits must be a 3D array")
                        patch_left_in_logits = int(patch_x * logits_shape[1] / num_patches_x)
                        patch_top_in_logits = int(patch_y * logits_shape[0] / num_patches_y)
                        patch_width_in_logits = int(logits_shape[1] / num_patches_x)
                        patch_height_in_logits = int(logits_shape[0] / num_patches_y)
                        patch_logits = logits[patch_left_in_logits:patch_left_in_logits + patch_width_in_logits, patch_top_in_logits:patch_top_in_logits + patch_height_in_logits, :]
                        new_prediction_record['logits'] = patch_logits.tolist()
                    
                    # Embeddings
                    feature_vector = pg_session.query(FeatureVector).filter(
                        FeatureVector.prediction == record['id'],
                        FeatureVector.type == 'EMBEDDINGS',
                        FeatureVector.model_name == record['model_name']
                    ).first()
                    if feature_vector is not None:
                        embeddings = decode_to_np_array(feature_vector.encoded_value)
                        embeddings_shape = embeddings.shape
                        patch_left_in_logits = int(patch_x * embeddings_shape[1] / num_patches_x)
                        patch_top_in_logits = int(patch_y * embeddings_shape[0] / num_patches_y)
                        patch_width_in_logits = int(embeddings_shape[1] / num_patches_x)
                        patch_height_in_logits = int(embeddings_shape[0] / num_patches_y)
                        if len(embeddings_shape) == 3:
                            patch_embeddings = embeddings[patch_left_in_logits:patch_left_in_logits + patch_width_in_logits, patch_top_in_logits:patch_top_in_logits + patch_height_in_logits, :]
                        elif len(embeddings_shape) == 2:
                            patch_embeddings = embeddings[patch_left_in_logits:patch_left_in_logits + patch_width_in_logits, patch_top_in_logits:patch_top_in_logits + patch_height_in_logits]
                        else:
                            raise Exception("Embeddings must be a 2D or 3D array")
                        new_prediction_record['embeddings'] = patch_embeddings.tolist()
                    
                    new_prediction_records.append(new_prediction_record)
        else:
            raise Exception(f"Task_type {task_type} does not support patch sampling")
        
        return new_prediction_records

    def preprocess_groundtruth(self, groundtruth):

        return []
