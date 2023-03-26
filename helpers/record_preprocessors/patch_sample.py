from sqlalchemy.orm import object_session

from schemas.pgsql import models
from ..eventprocessor.utils import encode_np_array, decode_to_np_array

from helpers.record_preprocessors.record_preprocessor import RecordPreprocessor

Prediction = models.prediction.Prediction
GroundTruth = models.groundtruth.GroundTruth
FeatureVector = models.feature_vector.FeatureVector

def slice_nparray(nparray, normalized_roi, accept_ndims=[2, 3]):
    if nparray is None:
        return None

    if nparray.ndim == 2 and 2 in accept_ndims:
        top = int(normalized_roi['top'] * nparray.shape[0])
        left = int(normalized_roi['left'] * nparray.shape[1])
        height = int(normalized_roi['height'] * nparray.shape[0])
        width = int(normalized_roi['width'] * nparray.shape[1])
        # Adjust the height and width to make sure the last index is also included, otherwise `int()` might round down and exclude the last index.
        if top + height + 1 == nparray.shape[0]:
            height += 1
        if left + width + 1 == nparray.shape[1]:
            width += 1
        return nparray[top:top+height, left:left+width]
    elif nparray.ndim == 3 and 3 in accept_ndims:
        top = int(normalized_roi['top'] * nparray.shape[1])
        left = int(normalized_roi['left'] * nparray.shape[2])
        height = int(normalized_roi['height'] * nparray.shape[1])
        width = int(normalized_roi['width'] * nparray.shape[2])
        # Adjust the height and width to make sure the last index is also included, otherwise `int()` might round down and exclude the last index.
        if top + height + 1 == nparray.shape[1]:
            height += 1
        if left + width + 1 == nparray.shape[2]:
            width += 1
        
        return nparray[:, top:top+height, left:left+width]
    else:
        raise Exception(f"Unsupported nparray dimension: {nparray.ndim}")

class PatchSamplePreprocessor(RecordPreprocessor):
    def __init__(self, size, **kwargs):
        self.size = size

    # A function that slices the given datapoint into patch records for datapoints.
    # Each record is a patch, with the same metadata as the parent datapoint, but with
    # metadata.roi = [top, left, height, width]
    # Each record also has its predictions and groundtruths with the following associated objects sliced up:
    # - segmentation_class_mask
    # - logits
    # - embeddings
    def preprocess_to_record(self, parent_datapoint):
        orm_session = object_session(parent_datapoint)
        num_patches_x = self.size[0]
        num_patches_y = self.size[1]

        new_datapoint_records = []
        for patch_x in range(0, num_patches_x):
            for patch_y in range(0, num_patches_y):
                new_datapoint_record = parent_datapoint._asdict()
                del new_datapoint_record['id']
                new_datapoint_records.append(new_datapoint_record)

                # ROI
                new_datapoint_record['metadata'] = new_datapoint_record.get('metadata', {})
                normalized_roi = {
                    'top': patch_x / num_patches_x,
                    'left': patch_y / num_patches_y,
                    'height': 1 / num_patches_x,
                    'width': 1 / num_patches_y
                }
                new_datapoint_record['metadata']['normalized_roi'] = normalized_roi

                # Predictions
                new_datapoint_record['predictions'] = []
                for parent_prediction in orm_session.query(Prediction).filter(Prediction.datapoint == parent_datapoint.id).all():
                    new_prediction_record = parent_prediction._asdict()
                    del new_prediction_record['id']
                    del new_prediction_record['datapoint']
                    new_datapoint_record['predictions'].append(new_prediction_record)
                    # Segmentation class mask
                    segmentation_class_mask = decode_to_np_array(parent_prediction.encoded_segmentation_class_mask)
                    new_prediction_record['segmentation_class_mask'] = slice_nparray(segmentation_class_mask, normalized_roi, accept_ndims=[2]).tolist()

                    # Logits
                    feature_vector = orm_session.query(FeatureVector).filter(
                        FeatureVector.prediction == parent_prediction.id,
                        FeatureVector.type == 'LOGITS'
                    ).first()
                    if feature_vector is not None:
                        logits = decode_to_np_array(feature_vector.encoded_value)
                        new_prediction_record['logits'] = slice_nparray(logits, normalized_roi).tolist()

                    # Embeddings
                    feature_vector = orm_session.query(FeatureVector).filter(
                        FeatureVector.prediction == parent_prediction.id,
                        FeatureVector.type == 'EMBEDDINGS'
                    ).first()
                    if feature_vector is not None:
                        embeddings = decode_to_np_array(feature_vector.encoded_value)
                        new_prediction_record['embeddings'] = slice_nparray(embeddings, normalized_roi).tolist()

                # Groundtruths
                new_datapoint_record['groundtruths'] = []
                for parent_groundtruth in orm_session.query(GroundTruth).filter(GroundTruth.datapoint == parent_datapoint.id).all():
                    new_groundtruth_record = parent_groundtruth._asdict()
                    del new_groundtruth_record['id']
                    del new_groundtruth_record['datapoint']
                    new_datapoint_record['groundtruths'].append(new_groundtruth_record)

                    # Segmentation class mask
                    segmentation_class_mask = decode_to_np_array(parent_groundtruth.encoded_segmentation_class_mask)
                    new_groundtruth_record['segmentation_class_mask'] = slice_nparray(segmentation_class_mask, normalized_roi, accept_ndims=[2]).tolist()

        return new_datapoint_records
