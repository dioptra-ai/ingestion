
from schemas.pgsql import models

from app import process_records
from ..eventprocessor.utils import decode_to_np_array, encode_np_array

from . import RecordPreprocessor

Datapoint = models.datapoint.Datapoint
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

    def process_datapoint(self, parent_datapoint, pg_session):
        logs = []
        preprocessed_records = self._preprocess_datapoint(parent_datapoint, pg_session)

        logs += process_records(preprocessed_records, parent_datapoint.organization_id, parent_pg_session=pg_session)

        self._post_process_datapoint(parent_datapoint, pg_session)

        return logs

    # A function that slices the given datapoint into patch records for datapoints.
    # Each record is a patch, with the same metadata as the parent datapoint, but with
    # metadata.roi = [top, left, height, width]
    # Each record also has its predictions and groundtruths with the following associated objects sliced up:
    # - segmentation_class_mask
    # - logits
    # - embeddings
    def _preprocess_datapoint(self, parent_datapoint, pg_session):
        num_patches_x = self.size[0]
        num_patches_y = self.size[1]

        new_datapoint_records = []
        for patch_x in range(0, num_patches_x):
            for patch_y in range(0, num_patches_y):
                new_datapoint_record = parent_datapoint._asdict()
                new_datapoint_record['parent_datapoint'] = parent_datapoint.id
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

                # Embeddings
                feature_vector = pg_session.query(FeatureVector).filter(
                    FeatureVector.datapoint == parent_datapoint.id,
                    FeatureVector.type == 'EMBEDDINGS'
                ).first()
                if feature_vector is not None:
                    embeddings = decode_to_np_array(feature_vector.encoded_value)
                    new_datapoint_record['embeddings'] = slice_nparray(embeddings, normalized_roi).tolist()

                # Predictions
                new_datapoint_record['predictions'] = []
                for parent_prediction in pg_session.query(Prediction).filter(Prediction.datapoint == parent_datapoint.id).all():
                    new_prediction_record = parent_prediction._asdict()
                    del new_prediction_record['id']
                    del new_prediction_record['datapoint']
                    new_datapoint_record['predictions'].append(new_prediction_record)
                    # Segmentation class mask
                    if parent_prediction.encoded_segmentation_class_mask is not None:
                        segmentation_class_mask = decode_to_np_array(parent_prediction.encoded_segmentation_class_mask)
                        new_prediction_record['segmentation_class_mask'] = slice_nparray(segmentation_class_mask, normalized_roi, accept_ndims=[2]).tolist()

                    # Logits
                    feature_vector = pg_session.query(FeatureVector).filter(
                        FeatureVector.prediction == parent_prediction.id,
                        FeatureVector.type == 'LOGITS'
                    ).first()
                    if feature_vector is not None:
                        logits = decode_to_np_array(feature_vector.encoded_value)
                        new_prediction_record['logits'] = slice_nparray(logits, normalized_roi).tolist()

                    # Embeddings
                    feature_vector = pg_session.query(FeatureVector).filter(
                        FeatureVector.prediction == parent_prediction.id,
                        FeatureVector.type == 'EMBEDDINGS'
                    ).first()
                    if feature_vector is not None:
                        embeddings = decode_to_np_array(feature_vector.encoded_value)
                        new_prediction_record['embeddings'] = slice_nparray(embeddings, normalized_roi).tolist()

                # Groundtruths
                new_datapoint_record['groundtruths'] = []
                for parent_groundtruth in pg_session.query(GroundTruth).filter(GroundTruth.datapoint == parent_datapoint.id).all():
                    new_groundtruth_record = parent_groundtruth._asdict()
                    del new_groundtruth_record['id']
                    del new_groundtruth_record['datapoint']
                    new_datapoint_record['groundtruths'].append(new_groundtruth_record)

                    # Segmentation class mask
                    if parent_groundtruth.encoded_segmentation_class_mask is not None:
                        segmentation_class_mask = decode_to_np_array(parent_groundtruth.encoded_segmentation_class_mask)
                        new_groundtruth_record['segmentation_class_mask'] = slice_nparray(segmentation_class_mask, normalized_roi, accept_ndims=[2]).tolist()

        return new_datapoint_records

    # A function that takes the given datapoint and slices up its predictions and groundtruths again.
    # This time, the slices are evenly sized, unlike what was used for the metrics and masks calculations.
    def _post_process_datapoint(self, parent_datapoint, pg_session):
        print('post-processing datapoint')

        # Fetch all children datapoints, and reslice their predictions and groundtruths but this time
        # with evenly sized slices.
        children_datapoints = pg_session.query(Datapoint).filter(Datapoint.parent_datapoint == parent_datapoint.id).all()
        for child_datapoint in children_datapoints:
            # ROI
            normalized_roi = child_datapoint.metadata_.get('normalized_roi', None)
            if normalized_roi is None:
                continue

            # Embeddings
            parent_vector = pg_session.query(FeatureVector.encoded_value, FeatureVector.id).filter(
                FeatureVector.datapoint == parent_datapoint.id,
                FeatureVector.type == 'EMBEDDINGS'
            ).first()
            if parent_vector is not None:
                embeddings = decode_to_np_array(parent_vector.encoded_value)
                pg_session.query(FeatureVector).filter(FeatureVector.datapoint == child_datapoint.id).update({
                    FeatureVector.encoded_value: encode_np_array(slice_nparray(embeddings, normalized_roi, force_even_slices=True))
                })

            # Predictions
            for child_prediction in pg_session.query(Prediction.model_name, Prediction.id).filter(Prediction.datapoint == child_datapoint.id).all():
                # Find the matching parent prediction.
                parent_prediction = pg_session.query(Prediction.id, Prediction.encoded_segmentation_class_mask).filter(
                    Prediction.datapoint == parent_datapoint.id,
                    Prediction.model_name == child_prediction.model_name
                ).first()
                if parent_prediction is None:
                    raise Exception(f"Could not find parent prediction for child prediction {child_prediction.id}")
                
                # Segmentation class mask
                if parent_prediction.encoded_segmentation_class_mask is not None:
                    # Update the child prediction's segmentation class mask.
                    segmentation_class_mask = decode_to_np_array(parent_prediction.encoded_segmentation_class_mask)
                    pg_session.query(Prediction).filter(Prediction.id == child_prediction.id).update({
                        Prediction.encoded_segmentation_class_mask: encode_np_array(slice_nparray(segmentation_class_mask, normalized_roi, accept_ndims=[2], force_even_slices=True))
                    })
                
                # Logits
                parent_vector = pg_session.query(FeatureVector.encoded_value).filter(
                    FeatureVector.prediction == parent_prediction.id,
                    FeatureVector.type == 'LOGITS'
                ).first()
                if parent_vector is not None:
                    logits = decode_to_np_array(parent_vector.encoded_value)
                    pg_session.query(FeatureVector).filter(FeatureVector.prediction == child_prediction.id).update({
                        FeatureVector.encoded_value: encode_np_array(slice_nparray(logits, normalized_roi, force_even_slices=True))
                    })

                # Embeddings
                parent_vector = pg_session.query(FeatureVector.encoded_value).filter(
                    FeatureVector.prediction == parent_prediction.id,
                    FeatureVector.type == 'EMBEDDINGS'
                ).first()
                if parent_vector is not None:
                    embeddings = decode_to_np_array(parent_vector.encoded_value)
                    pg_session.query(FeatureVector).filter(FeatureVector.prediction == child_prediction.id).update({
                        FeatureVector.encoded_value: encode_np_array(slice_nparray(embeddings, normalized_roi, force_even_slices=True))
                    })
                
            # Groundtruths
            for child_groundtruth in pg_session.query(GroundTruth.task_type, GroundTruth.id).filter(GroundTruth.datapoint == child_datapoint.id).all():
                # Find the matching parent groundtruth.
                parent_groundtruth = pg_session.query(GroundTruth.encoded_segmentation_class_mask).filter(
                    GroundTruth.datapoint == parent_datapoint.id,
                    GroundTruth.task_type == child_groundtruth.task_type
                ).first()
                if parent_groundtruth is None:
                    raise Exception(f"Could not find parent groundtruth for child groundtruth {child_groundtruth.id}")
                
                # Segmentation class mask
                if parent_groundtruth.encoded_segmentation_class_mask is not None:
                    # Update the child groundtruth's segmentation class mask.
                    segmentation_class_mask = decode_to_np_array(parent_groundtruth.encoded_segmentation_class_mask)
                    pg_session.query(GroundTruth).filter(GroundTruth.id == child_groundtruth.id).update({
                        GroundTruth.encoded_segmentation_class_mask: encode_np_array(slice_nparray(segmentation_class_mask, normalized_roi, accept_ndims=[2], force_even_slices=True))
                    })
