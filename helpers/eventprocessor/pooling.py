import numpy as np
import cv2


def pool2D(arr,
           kernel=(2, 2),
           stride=(2, 2),
           func=np.nanmax,
           ):

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    kernel += (1,)
    stride += (1,)

    # check inputs
    assert arr.ndim == 3
    assert len(kernel) == 3

    # create array with lots of padding around it, from which we grab stuff (could be more efficient, yes)
    arr_padded_shape = arr.shape + 2 * np.array(kernel)
    arr_padded = np.zeros(arr_padded_shape, dtype=arr.dtype) * np.nan
    arr_padded[
        kernel[0]:kernel[0] + arr.shape[0],
        kernel[1]:kernel[1] + arr.shape[1],
        kernel[2]:kernel[2] + arr.shape[2],
    ] = arr

    # create temporary array, which aggregates kernel elements in last axis
    size_x = 1 + (arr.shape[0]-1) // stride[0]
    size_y = 1 + (arr.shape[1]-1) // stride[1]
    size_z = 1 + (arr.shape[2]-1) // stride[2]
    size_kernel = np.prod(kernel)
    arr_tmp = np.empty((size_x, size_y, size_z, size_kernel), dtype=arr.dtype)

    # fill temporary array
    kx_center = (kernel[0] - 1) // 2
    ky_center = (kernel[1] - 1) // 2
    kz_center = (kernel[2] - 1) // 2
    idx_kernel = 0
    for kx in range(kernel[0]):
        dx = kernel[0] + kx - kx_center
        for ky in range(kernel[1]):
            dy = kernel[1] + ky - ky_center
            for kz in range(kernel[2]):
                dz = kernel[2] + kz - kz_center
                arr_tmp[:, :, :, idx_kernel] = arr_padded[
                                                   dx:dx + arr.shape[0]:stride[0],
                                                   dy:dy + arr.shape[1]:stride[1],
                                                   dz:dz + arr.shape[2]:stride[2],
                                               ]
                idx_kernel += 1

    # perform pool function
    arr_final = func(arr_tmp, axis=-1)
    return arr_final

def roi_pooling(mat, proposals, ksize):
    """
        Given feature layers and a list of proposals, it returns pooled
        respresentations of the proposals. Proposals are scaled by scaling factor
        before pooling.

        Args:
            mat (np.Array): Feature layer of size (height, width, num_channels)
            proposals (list of np.Array): Each element of the list represents a bounding
            box as (top,left,height,width) in relative coordinates

        Returns:
            np.Array: Shape len(proposals), channels, self.output_size, self.output_size
    """

    if not isinstance(mat, np.ndarray):
        mat = np.array(mat)

    if not isinstance(proposals, np.ndarray):
        proposals = np.array(proposals)

    mat = mat.astype(dtype=np.float32)

    height, width, num_channels = mat.shape

    scaled_proposals = np.zeros_like(proposals, dtype=np.int)
    scaled_proposals[:, 0] = np.ceil(proposals[:, 0] * height).astype(int) # Top
    scaled_proposals[:, 1] = np.ceil(proposals[:, 1] * width).astype(int) # Left
    scaled_proposals[:, 2] = np.ceil(proposals[:, 2] * height).astype(int) # Height
    scaled_proposals[:, 3] = np.ceil(proposals[:, 3] * width).astype(int) # With

    res = np.zeros((len(scaled_proposals), ksize,
                    ksize, num_channels))
    for idx in range(len(scaled_proposals)):
        extracted_feat = mat[
            scaled_proposals[idx, 0]:scaled_proposals[idx, 0] + scaled_proposals[idx, 2],
            scaled_proposals[idx, 1]:scaled_proposals[idx, 1] + scaled_proposals[idx, 3],
            :]

        if 0 not in extracted_feat.shape:
            res[idx] = cv2.resize(
                extracted_feat,
                dsize=(ksize, ksize),
                interpolation=cv2.INTER_LINEAR)

    return res
