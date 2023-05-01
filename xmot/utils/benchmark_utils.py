import glob
import re
from typing import Dict, List
from xmot.digraph.parser import parse_pascal_xml
from xmot.digraph.particle import Particle


def load_labels(data_dir) -> Dict[int, Dict[int, List[Particle]]]:
    """
    returns a dict of dict, the inner dict of which is {"<frame_id>" : List of particles}, and the
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
        if video_id not in ret:
            ret[video_id] = {image_id: particles} # Add a new video to the dict.
        else:
            ret[video_id][image_id] = particles # Add a new image to the existing video.
    
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