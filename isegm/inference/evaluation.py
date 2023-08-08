from time import time

import numpy as np
import torch
import os
from isegm.inference import utils
from isegm.inference.clicker import Clicker
import shutil
import cv2
from isegm.utils.distributed import get_rank, synchronize, get_world_size, all_gather
import itertools

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

def evaluate_dataset_dist(dataset, predictor, vis = True, vis_path = './experiments/vis_val/',**kwargs):
    all_ious = []
    rank, world_size = get_rank(), get_world_size()
    if vis:
        save_dir =  vis_path + dataset.name + '/'
        #save_dir = '/home/admin/workspace/project/data/logs/'+ dataset.name + '/'
        if rank == 0:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
    else:
        save_dir = None
    start_time = time()
    if rank == 0:
        pbar = tqdm(total=len(dataset), unit='image')
    for index in range(rank, len(dataset), world_size):
        sample = dataset.get_sample(index)
        _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, sample.init_mask, predictor,
                                            sample_id=index, vis= vis, save_dir = save_dir,
                                            index = index, **kwargs)
        all_ious.append(sample_ious)
        if rank == 0:
            for _ in range(world_size):
                pbar.update(1)
                pbar.set_description(f'Test {index+world_size}')
    
    synchronize()
    if rank == 0:
        pbar.close()
    all_ious = all_gather(all_ious)
    all_ious = list(itertools.chain(*all_ious))
    end_time = time()
    elapsed_time = end_time - start_time
    return all_ious, elapsed_time

def evaluate_dataset(dataset, predictor, vis = True, vis_path = './experiments/vis_val/',**kwargs):
    all_ious = []
    if vis:
        save_dir =  vis_path + dataset.name + '/'
        #save_dir = '/home/admin/workspace/project/data/logs/'+ dataset.name + '/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        save_dir = None

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, sample.init_mask, predictor,
                                            sample_id=index, vis= vis, save_dir = save_dir,
                                            index = index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time
    return all_ious, elapsed_time

def Progressive_Merge(pred_mask, previous_mask, y, x):
    diff_regions = np.logical_xor(previous_mask, pred_mask)
    num, labels = cv2.connectedComponents(diff_regions.astype(np.uint8))
    label = labels[y,x]
    corr_mask = labels == label
    if previous_mask[y,x] == 1:
        progressive_mask = np.logical_and( previous_mask, np.logical_not(corr_mask))
    else:
        progressive_mask = np.logical_or( previous_mask, corr_mask)
    return progressive_mask


def evaluate_sample(image, gt_mask, init_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, vis = True, save_dir = None, index = 0,  callback=None,
                    progressive_mode = True,
                    ):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    prev_mask = pred_mask
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)
        if init_mask is not None:
            predictor.set_prev_mask(init_mask)
            pred_mask = init_mask
            prev_mask = init_mask
            num_pm = 0
        else:
            num_pm = 999
        
        pred_masks = []
        for click_indx in range(max_clicks):
            vis_pred = prev_mask
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if progressive_mode:
                clicks = clicker.get_clicks()
                if len(clicks) >= num_pm: 
                    last_click = clicks[-1]
                    last_y, last_x = last_click.coords[0], last_click.coords[1]
                    pred_mask = Progressive_Merge(pred_mask, prev_mask,last_y, last_x)
                    predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask,0),0)
            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)
            prev_mask = pred_mask
            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break
        
        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs