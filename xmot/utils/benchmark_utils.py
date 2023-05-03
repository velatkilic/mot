import glob
import re
import numpy as np
import cv2 as cv
import json
from typing import Dict, List
from xmot.digraph.parser import parse_pascal_xml
from xmot.digraph.particle import Particle
from typing import List, Tuple, Dict

def load_labels(data_dir) -> Dict[int, Dict[int, List[int]]]:
    """
    Return a dict of dict, the inner dict of which is {"<frame_id>" : List of bbox}, and the
    outer dict of which uses video id as the key.

    data_dir assumes the label_studio format. It should contains two subfolder: "Annotations",
    and "images".
    """
    xmls = glob.glob("{:s}/Annotations/*.xml".format(data_dir))
    ret = {} # Return value
    for xml in xmls:
        particles, img_file_name = parse_pascal_xml(xml)
        obj = re.match(".*_([0-9]+)_([a-zA-Z]*)([0-9]+)\.([a-zA-Z]+)", img_file_name)
        video_id = int(obj.group(1))
        image_id = int(obj.group(3)) # frame_id
        bbox = [p.get_bbox_torch() for p in particles]
        if video_id not in ret:
            ret[video_id] = {image_id: bbox} # Add a new video to the dict.
        else:
            ret[video_id][image_id] = bbox # Add a new image to an existing video.
    
    return ret

def iou(bbox_1, bbox_2) -> float:
    """
    Calculate the "intersection over union" for a pair of bounding boxes.

    PASCAL VOC has two thresholds of 0.5 and 0.75.
    """
    # Area of intersection.
    x1 = max(bbox_1[0], bbox_2[0])
    y1 = max(bbox_1[1], bbox_2[1])
    x2 = min(bbox_1[2], bbox_2[2])
    y2 = min(bbox_1[3], bbox_2[3])

    # Area of union, the inverse of intersection
    x3 = min(bbox_1[0], bbox_2[0])
    y3 = min(bbox_1[1], bbox_2[1])
    x4 = max(bbox_1[2], bbox_2[2])
    y4 = max(bbox_1[3], bbox_2[3])

    # Include the boundary point itself.
    area_intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_union = (x4 - x3 + 1) * (y4 - y3 + 1)

    return area_intersection / area_union    

def intersect(bbox_1, bbox_2) -> List[int]:
    """
    Get the intersection of two bboxes. If only intersect at a point or a line, return the bbox
    describing that point / line.
    
    Return [0, 0, 0, 0] only if there's no intersection.
    """
    x1 = max(bbox_1[0], bbox_2[0])
    y1 = max(bbox_1[1], bbox_2[1])
    x2 = min(bbox_1[2], bbox_2[2])
    y2 = min(bbox_1[3], bbox_2[3])
    if x2 < x1 or y2 < y1:
        return [0, 0, 0, 0]

    return [x1, y1, x2, y2]

def union(bbox_1, bbox_2) -> List[int]:
    x1 = min(bbox_1[0], bbox_2[0])
    y1 = min(bbox_1[1], bbox_2[1])
    x2 = max(bbox_1[2], bbox_2[2])
    y2 = max(bbox_1[3], bbox_2[3])
    return [x1, y1, x2, y2]

def draw_bbox_with_id(shape, bbox: List[List[int]], padding=30, fontScale=0.75) -> np.ndarray:
    """
    Draw the list of bbox annotated with id on a white image. The image is padded to
    accomodate all the ids in case bboxes are close to boundaries of the image.

    Attributes:
        fontScale:  float   Font size.
    """
    img = np.full(shape, 255, dtype=np.uint8)
    # Add a padding to see the texts that are outside of the boundary.
    img = cv.copyMakeBorder(img, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=255)
    # Coordinates (x, y) are (column-index, row-index)
    img = cv.rectangle(img, (padding, padding), np.array((shape[1], shape[0])) + padding, color=0)
    for i, bbox in enumerate(bbox):
        img = cv.putText(img, str(i), np.array((bbox[0], bbox[1])) + padding, 
                         cv.FONT_HERSHEY_SIMPLEX, fontScale, 0, 2, cv.LINE_AA)
        img = cv.circle(img, np.array((bbox[0], bbox[1])) + padding, radius=2, color=0, thickness=-1)
        img = cv.rectangle(img, np.array((bbox[0], bbox[1])) + padding,
                                np.array((bbox[2], bbox[3])) + padding, color=0)
    return img
        
def compare_bbox(gt_bbox: List[List[int]], pred_bbox: List[List[int]], threshold = 0.5,
                 allow_enclose = True):
    seen_gt = np.zeros(len(gt_bbox), dtype=bool)
    seen_pred = np.zeros(len(pred_bbox), dtype=bool)
    for i, gt in enumerate(gt_bbox):
        for j, pred in enumerate(pred_bbox):
            if seen_pred[j]:
                continue
            val = iou(gt, pred)
            if val > threshold:
                seen_pred[j] = True
                seen_gt[i] = True
                break

    if allow_enclose:
        for j, pred in enumerate(pred_bbox):
            if seen_pred[j]:
                continue
            for i, gt in enumerate(gt_bbox):
                if seen_gt[i]:
                    continue
                # The predicted bounding box is almost enclosed in the labelled
                # bounding box. It often happens when the particle is very small
                # and the hand-labeled bbox contains too much a margin.
                if iou(intersect(gt, pred), pred) > 0.8:
                    seen_gt[i] = True
                    seen_pred[j] = True
                    break

    return seen_gt, seen_pred

def get_binary_classification(seen_gt: List[bool], seen_pred: List[bool]):
    """
    Collect the count of true_positive, false_positive and false_negative from comparison 
    results.
    """
    # np.sum(seen_gt) sould be equal to np.sum(seen_pred)
    return {"true_positive": np.sum(seen_gt), \
            "false_positive": len(seen_gt) - np.sum(seen_gt), \
            "false_negative": len(seen_pred) - np.sum(seen_pred)}

def merge_classification_result(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dicts of the results of binary classification by adding values of the same key.
    """
    ret = {}
    for key in dict1:
        ret[key] = dict1[key] + dict2[key]
    return ret

def bbox_area(bbox: List[int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)

def save_prediction_bbox(file, dict_bbox: Dict[int, List[int]]):
    """
    The "dict_bbox" has the format:
        {<frame_id>: List[[x1, y1, x2, y2], ...], <frame_id> : List[[x1, y1, x2, y2], ...]}
    """
    with open(file, "w") as f:
        json.dump(dict_bbox, f)

def load_prediction_bbox(file) -> Dict[int, List[int]]:
    """
    The returned dict of bbox has the format:
        {<frame_id>: List[[x1, y1, x2, y2], ...], <frame_id> : List[[x1, y1, x2, y2], ...]}
    """
    with open(file, "r") as f:
        obj = json.load(f)
    ret = {}
    for key, val in obj.items(): # At loading, the key are readed as string.
        ret[int(key)] = val
    return ret

def save_prediction_cnt(file, predicted_cnt: Dict[int, List[np.ndarray]]):
    """
    The input dict of contours has the format of:
        {<frame_id>: List[np.ndarray, np.ndarray, ...], <frame_id> : List[np.ndarry, np.ndarray, ...]}
    
    Each np.ndarray in the list has the shape: (n, 1, 2), corresponding to the shape of contours in
    OpenCV.

    The output file should have extension ".npy", and it will be a binary file.
    """
    np.save(file, predicted_cnt) # The saved file is binary.

def load_prediction_cnt(file) -> Dict[int, List[np.ndarray]]:
    obj = np.load(file, allow_pickle=True)
    return obj[()]
