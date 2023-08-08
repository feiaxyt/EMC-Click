import math

import torch
import numpy as np

from isegm.inference.clicker import Click
from .base import BaseTransform
import torch.nn.functional as F


class ResizeTrans(BaseTransform):
    def __init__(self, l=480):
        super().__init__()
        self.crop_size = l

    def transform(self, image_nd, clicks_lists):
        image_height, image_width = image_nd.shape[2:4]
        self.image_height = image_height
        self.image_width  = image_width
        if isinstance(self.crop_size, (list, tuple)):
            self.crop_height = self.crop_size[0]
            self.crop_width = self.crop_size[1]
        else:
            scale = self.crop_size / min(image_height, image_width)
            self.crop_height = int(round(image_height * scale)) // 4 * 4
            self.crop_width = int(round(image_width * scale)) // 4 * 4
        image_nd_r = F.interpolate(image_nd, (self.crop_height, self.crop_width), mode = 'bilinear', align_corners=True)

        y_ratio = self.crop_height / image_height
        x_ratio = self.crop_width / image_width

        clicks_lists_resized = []
        for clicks_list in clicks_lists:
            clicks_list_resized = [click.copy(coords=(click.coords[0] * y_ratio, click.coords[1] * x_ratio  ))
                                   for click in clicks_list]
            clicks_lists_resized.append(clicks_list_resized)

        return image_nd_r, clicks_lists_resized

    def inv_transform(self, prob_map):
        new_prob_map = F.interpolate(prob_map, (self.image_height, self.image_width), mode='bilinear', align_corners=True)

        return new_prob_map

    def get_state(self):
        return self.x_offsets, self.y_offsets, self._counts

    def set_state(self, state):
        self.x_offsets, self.y_offsets, self._counts = state

    def reset(self):
        self.x_offsets = None
        self.y_offsets = None
        self._counts = None


def get_offsets(length, crop_size, min_overlap_ratio=0.2):
    if length == crop_size:
        return [0]

    N = (length / crop_size - min_overlap_ratio) / (1 - min_overlap_ratio)
    N = math.ceil(N)

    overlap_ratio = (N - length / crop_size) / (N - 1)
    overlap_width = int(crop_size * overlap_ratio)

    offsets = [0]
    for i in range(1, N):
        new_offset = offsets[-1] + crop_size - overlap_width
        if new_offset + crop_size > length:
            new_offset = length - crop_size

        offsets.append(new_offset)

    return offsets
