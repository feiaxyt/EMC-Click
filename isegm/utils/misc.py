import torch
import numpy as np
import os
from .log import get_root_logger


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, checkpoints_path, epoch=None, prefix='', verbose=True, multi_gpu=False):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.pth'
    else:
        checkpoint_name = f'{epoch:03d}.pth'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    logger = get_root_logger()
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')

    net = net.module if multi_gpu else net
    torch.save({'state_dict': net.state_dict(),
                'config': net._config}, str(checkpoint_path))


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax

def expand_bbox_with_random_ratio_and_bias(bbox, ratio_range=[1.2, 1.6], min_crop_size=None, bias = 0.3):
    ratio = np.random.randint(int(ratio_range[0]*10),int(ratio_range[1]*10))/10
    y1, y2, x1, x2 = bbox
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)

    if min_crop_size is not None:
        h = max(h,min_crop_size)
        w = max(w,min_crop_size)

    hmax, wmax = int(h * bias), int(w * bias)
    h_bias = np.random.randint(-hmax,hmax+1)
    w_bias = np.random.randint(-wmax,wmax+1)

    x1 = int(round(xc - w * 0.5)) + w_bias
    x2 = int(round(xc + w * 0.5)) + w_bias
    y1 = int(round(yc - h * 0.5)) + h_bias
    y2 = int(round(yc + h * 0.5)) + h_bias

    return y1,y2,x1,x2

def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (max(rmin, bbox[0]), min(rmax, bbox[1]),
            max(cmin, bbox[2]), min(cmax, bbox[3]))


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()

def load_filelist_commonset(root_dir, data_list, image_dir_name, ext='.jpg', save_file=True, override_list=False):
    file_path = os.path.join(root_dir, data_list)
    if os.path.exists(file_path) and not override_list:
        with open(file_path) as f:
            samples = [file.strip() for file in f.readlines()]
    else:
        samples = os.listdir(os.path.join(root_dir, image_dir_name))
        samples = sorted([sample for sample in samples if sample.endswith(ext)])
        if save_file:
            with open(file_path, 'w') as f:
                for sample in samples:
                    f.write(sample+'\n')

    return samples