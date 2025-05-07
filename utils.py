import argparse
import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a model without cross validation'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test for a single model'
    )
    parser.add_argument(
        '--infer',
        action='store_true',
        help='Infer for a single model'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU ID'
    )

    return parser.parse_args()


def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    """Convert binary mask to COCO RLE format"""
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array


def rle_to_binary_mask(rle, shape):
    """Convert RLE to binary mask

    Args:
        rle: RLE encoded mask
        shape: output shape (height, width)

    Returns:
        binary mask (numpy array)
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    starts = rle[0::2]
    lengths = rle[1::2]

    current_position = 0
    for start, length in zip(starts, lengths):
        current_position += start
        mask[current_position:current_position + length] = 1
        current_position += length

    return mask.reshape(shape)
