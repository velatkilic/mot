# -*- coding: utf-8 -*-
"""
Utility functions
"""

import math
import numpy as np
import cv2 as cv
import json
from typing import List, Tuple, Union
from sklearn.mixture import BayesianGaussianMixture
from PIL import Image
from copy import copy
from xmot.utils.image_utils import get_contour_center

def imosaic(img_list, size=None, gray=False):
    '''
    Create a mosaic image from a nested list of images

    Parameters
    ----------
    img_list : Nested list of images. Ex: [[img11,img12],[img21, img22]]
    gray: draw grayscale

    Returns
    -------
    Final image mosaic (H, W, C)

    '''

    # concat columns in every row
    rows = []
    for row in img_list:
        proc_list = []
        for temp in row:
            nc = len(temp.shape) # 2 for gray image, 3 for color

            # if color image but need to convert to gray
            if nc == 3 and gray:
                proc_list.append(cv.cvtColor(temp, cv.COLOR_BGR2GRAY))
            # if gray image but need to convert to color
            elif nc == 2 and not gray:
                proc_list.append(cv.cvtColor(temp, cv.COLOR_GRAY2BGR))
            # otherwise just append
            else:
                proc_list.append(temp)
        rows.append(np.concatenate(proc_list, axis=1))

    # concat all the rows
    output = np.concatenate(rows, axis=0)

    # resize to desired output size
    if size is not None:
        output = cv.resize(output, size)
    return output


def drawBox(img, bbox, color=(0,0,255)):
    '''
    Draw rectangular bounding box on a given image

    Parameters
    ----------
    img : Image
    bbox : List of rectangular bounding boxes to be drawn format (x1, y1, x2, y2)

    Returns
    -------
    img : Modified image

    '''
    for j in range(len(bbox)):
        x1,y1,x2,y2 = bbox[j]
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

    return img

def drawBlobs(img, blobs):
    for j in range(len(blobs)):
        x1,y1,x2,y2 = blobs[j].get_bbox(frame_id = -1) # Most recent frame
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        color       = blobs[j].color
        color       = (int(color[0]), int(color[1]), int(color[2]))
        if len(img.shape) != 3:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.rectangle(img, (x1, y1), (x2, y2), color, thickness = 2)
    return img

def writeBlobs(blobs, file, frameID):

    """
    Output positions and bbox dimension (Kalman filters' states) at each frame.
    """
    with open(file, "a") as f:
        for i in range(len(blobs)):
            #Logger.debug("Number of frames for this blob: {:d}".format(
            #    len(blobs[i].frames)))
            #x1,y1,x2,y2 = blobs[i].bbox
            #x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            #w = x2 - x1
            #h = y2 - y1
            x, y, w, h = np.round(blobs[i].state).astype(np.int32).tolist()
            idx = blobs[i].idx
            cnt_str = json.dumps(blobs[i].contours[frameID].tolist())
            #frames = blobs[i].frames  # Not used.
            f.write(("{:4d}; " * 5 + "{:4d}; {:s}\n").format(x, y, w, h, idx, frameID, cnt_str))
            #f.write("{:4d}".format(frameID))
            #f.write(",".join([str(frame) for frame in frames]))
            #f.write("\n")

def findClosestBox(x,y,bbox):
    '''
    Given coordinates x,y and a list of bounding boxes,
    find the box that is closest to x,y

    Parameters
    ----------
    x : (width) x coordinate of box search
    y : (height) y coordinate of box search
    bbox : List of bounding boxes each with (x1,y1,x2,y2) format

    Returns
    -------
    box : box that best matches (x,y)
    out : index of the output box

    '''
    dist = 10000
    box  = [0,0,0,0]
    idx  = 0
    out  = -1
    for b in bbox:
        cenx = (b[0] + b[2])/2.
        ceny = (b[1] + b[3])/2.
        dist_new = np.sqrt((x-cenx)**2 + (y-ceny)**2)
        if dist_new < dist:
            dist = dist_new
            box  = b
            out  = idx
        idx += 1
    return box, out

def cor2cen(bbox):
    '''
    Convert bounding box edge coordinates of the form (x1,y1,x2,y2) to
    center coordinates of the form (cenx, ceny, w, h)
    '''
    cenx = (bbox[0] + bbox[2])/2.
    ceny = (bbox[1] + bbox[3])/2.
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    return cenx,ceny,w,h

def cen2cor(cenx,ceny,w,h):
    '''
    Convert bounding box center coordinates of the form (cenx, ceny, w, h) to
    edge coordinates of the form (x1,y1,x2,y2)

    '''
    hw = w/2.
    hh = h/2.

    x1 = cenx - hw
    x2 = cenx + hw
    y1 = ceny - hh
    y2 = ceny + hh

    return x1,y1,x2,y2

def costMatrix(states, blobs, fixed_cost=80.):
    boxlen = len(states)
    blolen = len(blobs)

    # size of cost array twice the largest
    # that way every blob can be deleted and new bbox can be created
    length = 2*max(boxlen, blolen)
    cost = np.ones((length, length), dtype=np.float64) * fixed_cost

    # Calculate cost
    for i in range(boxlen):
        for j in range(blolen):
            cenx1,ceny1,w1,h1 = states[i]
            cenx2,ceny2,w2,h2 = blobs[j].state

            # eucledian distance
            cost[i][j] = np.sqrt( (cenx1 - cenx2)**2 + (ceny1 - ceny2)**2 ) \
                         + abs(w1 - w2) + abs(h1 - h2)

    return cost

def iou(bbox1, bbox2):
    '''
    Intersection over union for two bounding boxes

    see: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    Parameters
    ----------
    bbox1 : Bounding box (x1,y1,x2,y2)
    bbox2 : Bounding box (x1,y1,x2,y2)

    Returns
    -------
    Intersection over union [0, 1]

    '''
    x1 = max(bbox1[0],bbox2[0])
    y1 = max(bbox1[1],bbox2[1])
    x2 = min(bbox1[2],bbox2[2])
    y2 = min(bbox1[3],bbox2[3])

    aint = max(0, x2 - x1 + 1.) * max(0, y2 - y1 + 1.)
    a1   = (bbox1[2] - bbox1[0] + 1.) * (bbox1[3] - bbox1[1] + 1.)
    a2   = (bbox2[2] - bbox2[0] + 1.) * (bbox2[3] - bbox2[1] + 1.)

    iou = aint/(a1 + a2 - aint)

    return iou

def iom(bbox1, bbox2):
    '''
    Intersection over min for two bounding boxes

    see: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    Parameters
    ----------
    bbox1 : Bounding box (x1,y1,x2,y2)
    bbox2 : Bounding box (x1,y1,x2,y2)

    Returns
    -------
    Intersection over union [0, 1]

    '''
    x1 = max(bbox1[0],bbox2[0])
    y1 = max(bbox1[1],bbox2[1])
    x2 = min(bbox1[2],bbox2[2])
    y2 = min(bbox1[3],bbox2[3])

    aint = max(0, x2 - x1 + 1.) * max(0, y2 - y1 + 1.)
    a1   = (bbox1[2] - bbox1[0] + 1.) * (bbox1[3] - bbox1[1] + 1.)
    a2   = (bbox2[2] - bbox2[0] + 1.) * (bbox2[3] - bbox2[1] + 1.)

    iom = aint/min(a1, a2)

    return iom

def unionBlob(blob1, blob2):
    '''
    Merge two blobs

    Parameters
    ----------
    blob1 : Main blob
    blob2 : Blob that merges with the main

    Returns
    -------
    Main blob
    '''
    # Average posteior state
    blob1.kalm.statePost = (blob1.kalm.statePost + blob2.statePost())/2.

    # update bbox
    state      = blob1.kalm.statePost
    blob1.bbox = np.array(cen2cor(state[0],state[1],state[2],state[3]))

    # stub
    # modify frames and dead
    return blob1


def unionMask(mask1, mask2):
    # Returns the union of two binary input masks
    out = np.logical_or(mask1, mask2)
    return out.tolist()


def unionBox(bbox1, bbox2):
    # Returns the union of two input bounding boxes
    # bboxes have coordinates in (x1,y1,x2,y2) format

    # Min of top left corner
    x1 = np.minimum(bbox1[0], bbox2[0])
    y1 = np.minimum(bbox1[1], bbox2[1])

    # Max of bottom right corner
    x2 = np.maximum(bbox1[2], bbox2[2])
    y2 = np.maximum(bbox1[3], bbox2[3])

    return [x1, y1, x2, y2]

def intersect(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    return (max(0, x2 - x1 + 1.) * max(0, y2 - y1 + 1.)) > 0

def areaBbox(bbox):
    """
    Calculate the area of the rectangular defined by the bbox.
    """
    return (bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.)

def calcDist(bbox1,bbox2):
    cen1 = np.array([bbox1[0] + bbox1[2],bbox1[1] + bbox1[3]]) / 2.
    cen2 = np.array([bbox2[0] + bbox2[2],bbox2[1] + bbox2[3]]) / 2.
    dist = np.sqrt((cen1[0] - cen2[0])**2 + (cen1[1] - cen2[1])**2)
    return dist

def calcHW(bbox):
    h = abs(bbox[0] - bbox[2])
    w = abs(bbox[1] - bbox[3])
    return h, w

def calcDistMetric(bbox1, bbox2):
    dist  = calcDist(bbox1,bbox2)
    h1,w1 = calcHW(bbox1)
    h2,w2 = calcHW(bbox2)
    metr  = (dist**2)/(h1*w1) + (dist**2)/(h2*w2)
    return metr

def mergeBoxes(mask, bbox, speed, mag, max_speed, th_speed, th_dist, it):
    # create new lists
    mask_new = mask.tolist()
    bbox_new = bbox.tolist()
    speed_new = speed.tolist()

    for i in range(it):

        cursor_left = 0
        cursor_right = 0
        length = len(bbox_new)
        cnt = 0
        while (cursor_left < length):
            cursor_right = cursor_left + 1
            while (cursor_right < length):
                # relative speed
                speed1 = speed_new[cursor_left]
                speed2 = speed_new[cursor_right]
                if speed2 != 0 and speed1 != 0:
                    rel_speed = abs(2. * (speed2 - speed1) / (speed2 + speed1))
                else:
                    rel_speed = 0

                # get bounding boxes
                bbox1 = bbox_new[cursor_left]
                bbox2 = bbox_new[cursor_right]
                metric = calcDistMetric(bbox1, bbox2)

                # masks
                mask1 = mask_new[cursor_left]
                mask2 = mask_new[cursor_right]

                # merge
                ds = ((metric < th_dist) or intersect(bbox1, bbox2))
                if (rel_speed < th_speed) and ds:
                    # merge segmentation mask
                    mask_new[cursor_left] = unionMask(mask1, mask2)

                    # merge bounding boxes
                    bbox_new[cursor_left] = unionBox(bbox1, bbox2)

                    # calculate new speed
                    if max_speed != 0:
                        speed_new[cursor_left] = np.mean(mag[mask_new[cursor_left][0]]) / max_speed
                    else:
                        speed_new[cursor_left] = 0

                    # pop merged data from lists
                    mask_new.pop(cursor_right)
                    bbox_new.pop(cursor_right)
                    speed_new.pop(cursor_right)
                    length = length - 1  # adjust length of the list

                    cnt += 1  # keep track of total merged boxes
                else:
                    cursor_right = cursor_right + 1

                # # Debug
                # bbox_temp  = [bbox_new[cursor_left], bbox_new[cursor_right-1]]
                # speed_temp = [speed_new[cursor_left], speed_new[cursor_right-1]]
                # print("Relative speed: " + str(rel_speed))
                # print("Metric: " + str(metric))
                # img = drawBox(frame2[0:crop,0:crop,:].copy(), bbox_temp, speed_temp)
                # if (rel_speed < th_speed) and ds: print("merged")
                # cv.imshow("Under study", img)
                # k = cv.waitKey(0)& 0xff
                # if k ==27:
                #     cap.release()
                #     cv.destroyAllWindows()
                #     exit()

            cursor_left = cursor_left + 1

        # print("# of merged boxes: " + str(cnt))

    mask_new = np.asarray(mask_new)
    bbox_new = np.asarray(bbox_new)
    speed_new = np.asarray(speed_new)

    return mask_new, bbox_new, speed_new

def merge_with_BGMM(bbox, mask, mag):
    bbox_new = []
    mask_new = []

    z = np.zeros((len(bbox),1))
    bbox_wspeed = np.hstack((bbox, z))

    bgm = BayesianGaussianMixture(n_components=len(bbox), max_iter=10).fit(bbox_wspeed) # Dirichlet process
    asg = bgm.predict(bbox_wspeed) # cluster assignments
    uid = np.unique(asg) # unique clusters
    for idx in uid:
        indx = np.where(asg==idx)[0] # get indices for a given cluster assignment
        temp_box = bbox[indx[0],:] # set temp box to the first element
        temp_mask = mask[indx[0],...] # set temp mask to the first element
        for j in range(1,len(indx)):
            temp_box = unionBox(temp_box, bbox[indx[j],:])
            temp_mask = unionMask(temp_mask, mask[indx[j],...])
        bbox_new.append(temp_box)
        mask_new.append(temp_mask)

    mask_new = np.array(mask_new)
    bbox_new = np.array(bbox_new)
    return bbox_new, mask_new

def filterBbox(list_bbox: List[List[int]], list_cnt: List[np.ndarray]= None):
    """
    Postprocessing of list of bboxes and corresponding contours (in OpenCV format).

    Current filters:
    1. Check enclosement.
        If a smaller bbox is completed enclosed within a larger bbox, remove
        the smaller bbox from the list.
        TODO: Make the smaller bbox a bubble of the larger particle.
        TODO: Use the topology hierarchy of contours instead of bboxes.
    """
    if len(list_cnt) != len(list_bbox):
        print(f"Warning: Number of bbox and contours don't match. {len(list_cnt)} {len(list_bbox)}")

    to_remove = np.zeros(len(list_bbox), dtype=bool)

    for i in range(0, len(list_bbox)):
        if to_remove[i]:
            continue
        for j in range(i + 1, len(list_bbox)):
            if to_remove[j]:
                continue
            bbox_i = list_bbox[i]
            bbox_j = list_bbox[j]
            area_i = areaBbox(bbox_i)
            area_j = areaBbox(bbox_j)
            if iom(bbox_i, bbox_j) == 1:
                if area_i > area_j:
                    to_remove[j] = True
                else:
                    to_remove[i] = True

    ret_bbox = [list_bbox[i] for i in range(len(list_bbox)) if not to_remove[i]]
    ret_cnt = None
    if list_cnt != None:
        ret_cnt = [list_cnt[i] for i in range(len(list_bbox)) if not to_remove[i]]
    return ret_bbox, ret_cnt

def cnt_to_mask(cnt, height, width):
    """Convert one opencv contour into a pyTorch format mask.

    Pixels within the contour have value 255 (white) and pixels outside have value 0.

    Args:
        cnt:    One contour in opencv format, i.e. numpy array of Shape: [N, 1, 2] where N
                is the number of anchor points.
                Somehow, cv.findcontours() returns a tuple of contours
                and each contour is a 3D NumPy array with one redundant dimension in the
                middle.
                E.g. contours[0]:
                array([[[50, 50]],
                       [[50, 100]],
                       [[100, 100]],
                       [[100, 50]]], dtype=int32)

    Return:
        mask:   numpy.ndarray in the same shape of PyTorch predictions. [n, 1, height, width]
    """
    img = np.zeros((height, width), dtype=np.uint8)
    # 1. The function expect a list of contours. So I wrap the cnt in a list.
    # 2. 255 to indicate white
    # 3. thickness=-1: fill in the contour
    mask = cv.drawContours(img, [cnt], 0, (255), thickness=-1)

    mask = mask[np.newaxis, np.newaxis, :, :] # Make the mask the same format as that returned
                                              # from pyTorch. [n, 1, height, width]. n=1 here,
                                              # so the first dimension is 1.
    return mask

def mask_to_cnt(mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Get the contour from a binary mask.

    This function should be the conjugate function of cnt_to_mask(). We can use
    cv.matchShapes(cnt_1, cnt_2, cv.CONTOURS_MATCH_I1, 0.0) to check shape similarity of the origina
    and regenerated contour. A value of 0.0 means the contours are exactly the same.

    Args:
        mask: The mask of one particle. It should has the dimension (1, height, width)
              as those stored in Blob class of Kalman filter, or (1, 1, height, width)
              as the one directly generated from cnt_to_mask().
              When mask is an empty array, return an empty contour.

    Return:
        np.ndarray: One contour in the opencv format, with dimension (N, 1, 2) with N being the
                    number of anchor points of the contour.
        int:        image height (rows in numpy array). Not the height of the bbox of the contour.
        int:        image width (columns in numpy array)
    """
    if mask.size == 0: # Empty mask. No detected particle.
        return np.array([], dtype=np.int32).reshape(0, 1, 2), 0, 0

    mask = np.squeeze(mask)
    height, width = mask.shape
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8) * 255  # Make the mask as binary image.
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # There should be only one contour from the binary mask. Since background is black, there
    # isn't a contour of the whole image.
    return contours[0], height, width

def state_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get the Kalman filter state from a binary mask.

        Args:
        mask: The mask of one particle. It should has the dimension (1, height, width)
              as those stored in Blob class of Kalman filter, or (1, 1, height, width)
              as the one directly generated from cnt_to_mask().

    Return:
        int: x of centroid of the contour
        int: y of centroid of the contour
        int: width of the bbox of the contour
        int: height of the bbox of the contour
    """
    contour, img_height, img_width = mask_to_cnt(mask)
    x, y, w, h = cv.boundingRect(contour)
    centroid_x, centroid_y = get_contour_center(contour)
    return centroid_x, centroid_y, w, h


def opencv_to_pillow(cv_image) -> Image:
    """
    Converts an OpenCV image to a Pillow image.
    Automatically detects color or grayscale format.
    """
    if len(cv_image.shape) == 2:  # Grayscale
        return Image.fromarray(cv_image)
    elif len(cv_image.shape) == 3 and cv_image.shape[2] == 3:  # Color
        cv_rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        return Image.fromarray(cv_rgb_image)

def pillow_to_opencv(pil_image) -> np.ndarray[np.uint8]:
    """
    Converts a Pillow image to an OpenCV image.
    Automatically detects color or grayscale format.
    """
    np_image = np.array(pil_image)
    if len(np_image.shape) == 2:  # Grayscale
        return np_image
    elif len(np_image.shape) == 3 and np_image.shape[2] == 3:  # Color
        return cv.cvtColor(np_image, cv.COLOR_RGB2BGR)


def draw_dashed_rectangle(img: np.ndarray[np.uint8], top_left, bottom_right, color=(0, 0, 0),
                          dash_length=2, gap_length=3, thickness=1, inplace=True) -> np.ndarray[np.uint8]:
    """
    Draws a dashed rectangle on an OpenCV image.

    Parameters:
    - img: np.ndarray - OpenCV image (grayscale or color)
    - top_left: tuple - Coordinates of the top-left corner (x, y)
    - bottom_right: tuple - Coordinates of the bottom-right corner (x, y)
    - color: tuple - Color of the rectangle in BGR (for color image) or single value (for grayscale)
    - dash_length: int - Length of each dash
    - gap_length: int - Length of the gap between dashes
    - thickness: int - Thickness of the dashed line
    - inplace: bool - Draw on the original image or get a copy.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    img_result = copy.deepcopy(img) if not inplace else img

    def draw_dashed_line(start, end):
        """Draws a dashed line between two points."""
        line_length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        dashes = math.floor(line_length / (dash_length + gap_length))
        for i in range(dashes):
            #start_dash = (
            #    int(start[0] + (i * (dash_length + gap_length))),
            #    int(start[1] + (i * (dash_length + gap_length)))
            #)
            #end_dash = (
            #    int(start[0] + (i * (dash_length + gap_length)) + gap_length),
            #    int(start[1] + (i * (dash_length + gap_length)) + gap_length)
            #)
            start_dash = (
                int(start[0] + (end[0] - start[0]) * (i * (dash_length + gap_length) / line_length)),
                int(start[1] + (end[1] - start[1]) * (i * (dash_length + gap_length) / line_length))
            )
            end_dash = (
                int(start[0] + (end[0] - start[0]) * ((i * (dash_length + gap_length) + dash_length) / line_length)),
                int(start[1] + (end[1] - start[1]) * ((i * (dash_length + gap_length) + dash_length) / line_length))
            )

            cv.line(img_result, start_dash, end_dash, color, thickness)

    # Draw dashed lines for each side of the rectangle
    draw_dashed_line((x1, y1), (x2, y1))  # Top
    draw_dashed_line((x2, y1), (x2, y2))  # Right
    draw_dashed_line((x2, y2), (x1, y2))  # Bottom
    draw_dashed_line((x1, y2), (x1, y1))  # Left

    return img_result