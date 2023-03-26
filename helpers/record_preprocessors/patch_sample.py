from sqlalchemy.orm import object_session

from schemas.pgsql import models
from ..eventprocessor.utils import decode_to_np_array

from helpers.record_preprocessors.record_preprocessor import RecordPreprocessor

Prediction = models.prediction.Prediction
GroundTruth = models.groundtruth.GroundTruth
FeatureVector = models.feature_vector.FeatureVector

# Slices a numpy array into a patch based on the given normalized ROI.
# If force_even_slices=True, the array with be center-cropped so the patches are all the same size,
# otherwise the bordering patches might be larger depending on the divisibility of the array by the ROI.
def slice_nparray(nparray, normalized_roi, accept_ndims=[2, 3], force_even_slices=False):
    if nparray is None:
        raise Exception("Cannot slice None nparray.")

    dims = nparray.shape
    norm_roi_top = normalized_roi['top']
    norm_roi_left = normalized_roi['left']
    norm_roi_height = normalized_roi['height']
    norm_roi_width = normalized_roi['width']

    if nparray.ndim == 2 and 2 in accept_ndims:
        top = int(norm_roi_top * dims[0])
        left = int(norm_roi_left * dims[1])
        height = int(norm_roi_height * dims[0])
        width = int(norm_roi_width * dims[1])
        top_margin = int((dims[0] % (norm_roi_height * dims[0])) / 2)
        left_margin = int((dims[1] % (norm_roi_width * dims[1])) / 2)
        if force_even_slices:
            # If the array is not divisible by the ROI,
            # center-crop the array so the patches are all the same size if crop=True.
            nparray = nparray[top_margin: dims[0] - top_margin, left_margin: dims[1] - left_margin]
        else:
            # Else, if this is a border patch, extend the ROI with the appropriate indivisible amount.
            if top == 0: # Top border
                height += top_margin
            if left == 0: # Left border
                width += left_margin
            if top + 2 * height > dims[0]: # Bottom border (take 1 extra pixel in case of uneven margin height)
                height = dims[0] - top + 1
            if left + 2 * width > dims[1]: # Right border (take 1 extra pixel in case of uneven margin width)
                width = dims[1] - left + 1

        return nparray[top:top+height, left:left+width]
    elif nparray.ndim == 3 and 3 in accept_ndims:
        top = int(norm_roi_top * dims[1])
        left = int(norm_roi_left * dims[2])
        height = int(norm_roi_height * dims[1])
        width = int(norm_roi_width * dims[2])
        top_margin = int((dims[1] % (norm_roi_height * dims[1])) / 2)
        left_margin = int((dims[2] % (norm_roi_width * dims[2])) / 2)
        if force_even_slices:
            # If the array is not divisible by the ROI,
            # center-crop the array so the patches are all the same size if crop=True.
            nparray = nparray[:, top_margin: dims[1] - top_margin, left_margin: dims[2] - left_margin]
        else:
            # Else, if this is a border patch, extend the ROI with the appropriate indivisible amount.
            if top == 0:
                height += top_margin
            if left == 0:
                width += left_margin
            if top + 2 * height > dims[1]:
                height = dims[1] - top + 1
            if left + 2 * width > dims[2]:
                width = dims[2] - left + 1

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
                    if parent_prediction.encoded_segmentation_class_mask is not None:
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
                    if parent_groundtruth.encoded_segmentation_class_mask is not None:
                        segmentation_class_mask = decode_to_np_array(parent_groundtruth.encoded_segmentation_class_mask)
                        new_groundtruth_record['segmentation_class_mask'] = slice_nparray(segmentation_class_mask, normalized_roi, accept_ndims=[2]).tolist()

        return new_datapoint_records
