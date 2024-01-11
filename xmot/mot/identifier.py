import cv2 as cv
from xmot.mot.utils import drawBox, drawBlobs, writeBlobs, mergeBoxes, cnt_to_mask
from xmot.mot.kalman import MOT
from xmot.mot.detectors import DNN, GMM
from xmot.datagen.bead_gen import bead_data_to_file, BeadDatasetFile, collate_fn
from xmot.datagen.style_data_gen import StyleDatasetGen
import os
from xmot.logger import Logger
from torch.utils.data import DataLoader
import numpy as np
import sys

def identify(dset, imgOutDir, blobsOutFile, modelType="DNN", model=None, train_set=None, device="cuda:0", num_workers=0):
    """
    Identify particles using specified model.

    Attributes:
        dset          : Dataset  Instance of the video wrapper class Dataset.
        imgOutDir     : String  Output folder of video frames with detected bounding boxes.
        blobsOutFile  : String  Output file for info of each identified particle.
        modelType     : String  Type of detection model: DNN, GMM, or Canny.
        model         : String  Path to pre-trained model.
        device        : String  Device to be used for training and detecting. "cuda:0" or "cpu".
    """
    if modelType == "DNN":
        if train_set is None and model is None:
            # regular bead data
            filename = os.path.join(os.getcwd(), "train")
            try:
                os.mkdir(filename)
            except:
                Logger.warning("Folder of regular beads already exists! Overwriting existing data.")

            bead_data_to_file(filename)
            train_set = [filename]

            # style bead data
            filename = os.path.join(os.getcwd(), "train_style")
            train_set.append(filename)
            sdset = StyleDatasetGen(dset=dset, len=100)
            sdset.gen_dataset()

        # Object detection
        if model is None:
            model = DNN(device=device)
            for d in train_set:
                print("Train set: " + d)
                d = BeadDatasetFile(d)
                train_dataloader = DataLoader(d, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
                model.train(train_dataloader)

            # save model
            model.save_model()
        else:
            model = DNN(model, device=device)
    # We remodelled the GMM model entirely.
    #elif modelType == "GMM":
    #    model = GMM(dset.get_video_name(), dset.get_crop())

    # Tracking
    # Make directory
    kalman_dir = os.path.join(imgOutDir, "kalman")
    try:
        os.mkdir(kalman_dir)
    except:
        print("Folder already exists, overwriting contents ... ")

    Logger.detail("Detecting particles ...")
    
    for i in range(dset.length()-1):
        img = dset.get_img(i)
        # For return type, see:
        # https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
        # bbox: FloatTensor[N, 4]
        # mask: UInt8Tensor[num_particle, 1, height, width]
        bbox, mask = model.predict(img)
        
        if len(bbox) == 0: # No particles have been detected in this frame
            # Still write images out.
            cv.imwrite("{:s}/{:s}_{:d}.jpg".format(imgOutDir, modelType, i), img)
            cv.imwrite("{:s}/{:s}_{:d}.jpg".format(kalman_dir, modelType, i), img)
            continue

        # optical flow
        mask = mask.astype(bool)
        if len(img.shape)==3:
            cur = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            nxt = cv.cvtColor(dset.get_img(i+1),cv.COLOR_BGR2GRAY)
        else:
            cur = img
            nxt = dset.get_img(i+1)
        flow = cv.calcOpticalFlowFarneback(cur,nxt, None, pyr_scale=0.5, levels=5, 
                                           winsize=15, iterations=3, poly_n=5,
                                           poly_sigma=1.2, flags=0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        
        # Merge
        npar = len(bbox)
        
        # Average speed for each particle
        # cen   = np.zeros((npar,2)) # center position
        speed = np.zeros((npar,))  # average speed of entire particle mask
        for j in range(npar):
            # cen[j,:] = np.array([bbox[j,0] + bbox[j,2],          # x
            #                      bbox[j,1] + bbox[j,3]]) / 2.    # y
            # indxs = (xx-cen[j,0])**2 + (yy-cen[j,1])**2 <= r*r
            # indxs = np.logical_and(indxs, mask[j,...])
            speed[j] = np.mean(mag[mask[j,0,:,:]])
        
        # normalize speeds (useful for plotting later)
        max_speed = np.max(speed)
        if max_speed != 0:
            speed = speed / max_speed # Otherwise, all values are already 0.
        th_speed=0.2
        th_dist=2
        it =2
        mask, bbox, _ = mergeBoxes(mask, bbox, speed, mag, max_speed, th_speed, th_dist, it)
        
        # Draw bounding boxes
        cont = drawBox(img.copy(), bbox)

        # Show final image
        #cv.imshow("Frame", cont)
        cv.imwrite("{:s}/{:s}_{:d}.jpg".format(imgOutDir, modelType, i), cont)
        
        # Kalman tracking
        if "mot" not in locals(): # If have created a MOT object.
            mot = MOT(bbox, mask)
        else: # Use the existing MOT object.
            mot.step(bbox, mask)

        img_kalman = drawBlobs(img.copy(), mot.blobs)
        cv.imwrite("{:s}/{:s}_{:d}.jpg".format(kalman_dir, modelType, i), img_kalman)

        writeBlobs(mot.blobs, blobsOutFile, mot.cnt)

def identify_batch():
    """Detecting particles from all images first and then group particles into trajectories.
    
    TODO: 
    1. Move the GMM detection code into here as well.
    """
    pass

def build_trajectory_batch_GMM(dict_bbox, dict_cnt, images, kalman_dir="./", blobs_out="blobs.txt"):
    """A helper function to prepare variables from GMM detection to build_trajectory_batch().

    Sanity checks are performed to ensure there's one list of bboxes for each image, and the numbers
    of particles in an image are equal as in list of bboxes and masks.
    """
    # When there's no bounding boxes, there would still be an empty list.
    list_bboxes = [np.array(dict_bbox[i]) for i in sorted(dict_bbox.keys())]
    list_cnts = [dict_cnt[i] for i in sorted(dict_cnt.keys())]
    height = images[0].shape[0]
    width = images[0].shape[1]

    # Sanity checks. In case we encontour strange errors hard to trace to the true source of errors.
    if len(list_bboxes) != len(list_cnts) or len(list_bboxes) != len(images):
        Logger.error('KALMAN: Number of images not equal in list of bboxes, contours and images! '\
                     + f'{len(list_bboxes)} {len(list_cnts)} {len(images)}')
        sys.exit(1)

    equal = True
    for i in range(len(list_bboxes)):
        if len(list_bboxes[i]) != len(list_cnts[i]):
            Logger.error(f'KALMAN: Numbers of particles in frame-{i} in bbox and contours are not equal!' \
                         + f'{len(list_bboxes[i])} {len(list_cnts[i])}')
            equal = False

    if not equal:
        sys.exit(1)

    # Transfer each contour in opencv format into mask in pytorch format. Pixels of particles
    # have value 1, and the background has value 0.
    list_masks = []
    for cnts in list_cnts: # cnts: list of contours of particles in one image
        masks = []
        for cnt in cnts:
            mask = cnt_to_mask(cnt, height, width)
            masks.append(mask)
        
        if len(masks) > 0:
            # Each "mask" is a 4D numpy array. We need to concatenate them along the axis-0 to maintain
            # the mask variable of one image as a 4D array.
            list_masks.append(np.concatenate(masks, axis=0))
        else:
            list_masks.append(masks) # Add the empty list.

    # The "asarray" would raise errors becuase of inhomogeneous dimensions (i.e. different number
    # of particles for each image).
    # list_bboxes = np.asarray(list_bboxes)
    # list_masks = np.asarray(list_masks)
    
    build_trajectory_batch(list_bboxes, list_masks, images,
                           kalman_dir=kalman_dir, model_type="GMM", blobs_out=blobs_out)


def build_trajectory_batch(bboxes, masks, images, kalman_dir="./", model_type="GMM", blobs_out="blobs.txt"):
    """Build trajectories using the Kalman Filter with detections of all images.

    Args:
        bboxes  List[numpy.array]: List of bboxs in all images. Shape: [N_images, N_particles, 4]
                                   When there're no particles present in an image, an empty list should
                                   be in the list.
        masks   List[numpy.array]: List of masks of each particle in all images.
                                   Shape: [N_images, N_particles, 1, height, width]
                                   Areas of objects have value 1, and the background has value 0.
        images        numpy.array: List of images. Shape: [N_images, height, width, channel*]. For now,
                                   only pass in grayscale images since we primarily use this function
                                   with GMM detection results.
        blob_out              str: Path of file writing blobs to.
        
    Returns:
        None
    """
    if not os.path.exists(kalman_dir):
        os.makedirs(kalman_dir)

    for i, (bbox, mask, img) in enumerate(zip(bboxes, masks, images)):
        # Skip if no particles have been detected in the image.
        if len(bbox) == 0:
            Logger.detail(f'KALMAN: No particle has been detected in frame-{i}.')
            continue
        mask = mask.astype(bool) # 1 -> True, 0 -> False
        
        # optical flow
        if i < len(bboxes) - 1: # Not the last frame
            if len(img.shape)==3:
                cur = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                nxt = cv.cvtColor(images[i+1], cv.COLOR_BGR2GRAY)
            else:
                cur = img
                nxt = images[i+1]
            flow = cv.calcOpticalFlowFarneback(cur, nxt, None, pyr_scale=0.5, levels=5, 
                                               winsize=15, iterations=3, poly_n=5,
                                               poly_sigma=1.2, flags=0)
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            
            # Merge
            npar = len(bbox)
            
            # Average speed for each particle
            # cen   = np.zeros((npar,2)) # center position
            speed = np.zeros((npar,)) # 1D array with length npar
            for j in range(npar):
                # cen[j,:] = np.array([bbox[j,0] + bbox[j,2],          # x
                #                      bbox[j,1] + bbox[j,3]]) / 2.    # y
                # indxs = (xx-cen[j,0])**2 + (yy-cen[j,1])**2 <= r*r
                # indxs = np.logical_and(indxs, mask[j,...])
                speed[j] = np.mean(mag[mask[j,0,:,:]]) # average speed of entire particle mask
            
            # normalize speeds (useful for plotting later)
            max_speed = np.max(speed)
            if max_speed != 0:
                speed = speed / max_speed # Otherwise, all values are already 0.
            th_speed=0.2
            th_dist=2
            it =2
            mask, bbox, _ = mergeBoxes(mask, bbox, speed, mag, max_speed, th_speed, th_dist, it)
            
            # Draw bounding boxes
            # cont = drawBox(img.copy(), bbox)

            # Show final image
            # cv.imshow("Frame", cont)
            # cv.imwrite("{:s}/{:s}_{:d}.jpg".format(imgOutDir, model_type, i), cont)
        
        # Kalman tracking
        if "mot" not in locals(): # If have created a MOT object.
            mot = MOT(bbox, mask)
        else: # Use the existing MOT object.
            mot.step(bbox, mask)

        # Draw kalman filter blobs
        img_kalman = drawBlobs(img.copy(), mot.blobs)
        cv.imwrite("{:s}/{:s}_{:d}.jpg".format(kalman_dir, model_type, i), img_kalman)

        writeBlobs(mot.blobs, blobs_out, mot.cnt)