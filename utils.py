import os
import time
import logging
import numpy as np
from config import get_cfg_defaults

from skimage import measure, morphology

def get_hottest_point(heatmap):
    w, h = heatmap.shape
    flattened_heatmap = np.ndarray.flatten(heatmap)
    hottest_idx = np.argmax(flattened_heatmap)
    return np.flip(np.array(np.unravel_index(hottest_idx, [w, h])))


def prepare_config_output_and_logger(cfg_path, log_prefix):
    # get config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    # get directory to save log and model
    split_cfg_path = cfg_path.split("/")
    yaml_file_name = os.path.splitext(split_cfg_path[-1])[0]
    output_path = os.path.join('output', yaml_file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(log_prefix, time_str)
    log_path = os.path.join(output_path, log_file)
    save_model_path = os.path.join(output_path, yaml_file_name + "_model_" + time_str + ".pth")
    save_scaled_model_path = os.path.join(output_path, yaml_file_name + "_scaled_model.pth")

    # setup the logger
    logging.basicConfig(filename=log_path,
                        format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return cfg, logger, output_path, save_model_path, save_scaled_model_path

def makeGaussian(size: tuple, fwhm = 9, center: tuple = None) -> np.ndarray:
    """ Make a rectangle gaussian kernel.
    size is the length of a side of the rectangle
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    _size = np.max(size)        # the longger boarder
    x = np.arange(0, _size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = _size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)[:size[0], :size[1]]


def ExtractNLargestBlobs(binaryImage, numberToExtract):
    if numberToExtract < 0 :
        numberToExtract = -numberToExtract
    try:
        labels = measure.label(binaryImage, connectivity=1)
        if labels.max() > 1:
            regions = measure.regionprops(labels)
            allAreas = [regions[i].area for i in range(labels.max())]
            allAreas.sort(reverse=True)
            binaryImage = binaryImage.astype(np.bool)
            if numberToExtract > labels.max():
                offset=int(labels.max())-1
            else:
                offset=int(numberToExtract)-1
            dst=morphology.remove_small_objects(binaryImage, min_size=allAreas[offset],connectivity=1)
            dst.dtype = np.uint8
            return dst
        else :
            return binaryImage
    except:
        return False
    
def NormalizedTerm(norm_channels, output):
    np.sum(norm_channels * output)
    
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

def get_maxima(data, threshold = 0.01, neighborhood_size = 5):
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    cordinate = []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2    
        cordinate.append([x_center, y_center])
    return np.array(cordinate)

def save_pred_result(output_file, image_name, heatmap_stack):
    normalized_heatmaps = heatmap_stack[0] / np.max(heatmap_stack[0], axis=(1, 2), keepdims=True)
    predicted_landmark_positions = np.array([get_hottest_point(heatmap) for heatmap in normalized_heatmaps])
    line = "imagename" + image_name
    for i in range(len(predicted_landmark_positions)):
        line += f",{predicted_landmark_positions[i][0]},{predicted_landmark_positions[i][1]}"
    line += "\n"
    output_file.write(line)