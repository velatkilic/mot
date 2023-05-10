import click
import glob
import cv2 as cv
from pathlib import Path
import re
import numpy as np
import json

from xmot.digraph.parser import parse_pascal_xml
from xmot.analyzer.shapeDetector import detect_shape
from xmot.mot.detectors import DNN
from xmot.logger import Logger
from xmot.config import AREA_THRESHOLD, IMAGE_FILE_PATTERN

from xmot.utils.benchmark_utils import iou, intersect, union
from xmot.utils.benchmark_utils import compare_bbox, merge_confusion_matrix, get_confusion_matrix
from xmot.utils.benchmark_utils import calc_precision, calc_recall, draw_comparison
from xmot.utils.image_utils import subtract_brightfield_by_shifting as sbf_shifting


@click.command()
@click.argument("data_dir")
@click.argument("model")
@click.argument("outdir")
@click.option("--debug-output", type=str, default="", help="Folder to write debug images.")
@click.option("--area-threshold", type=int, default=AREA_THRESHOLD)
@click.option("--confidence-threshold", type=float, default=0.3, help="Confidence threshold of considering "\
              "a predicted bbox is actually a particle.")
@click.option("-sbf", "--subtract-brightfield", is_flag=True, help="Whether subtracting brightfield image to be subtracted.")
def benchmark(data_dir, model, outdir, area_threshold, debug_output, confidence_threshold, subtract_brightfield):
    """
    Collect all labelled data from .xml files in DATA_DIR and detect particles in corresponding
    images with DNN MODEL. Bencharmk results is written to OUTPUT.

    \b
    Note:
    1. This script assumes the XML files and corresponding images exist in "${data_dir}/Annotations"
       and "${data_dir}/images"
    2. In OUTPUT file, we will record statistics about the labelled data: total number of images,
       total number of particles (i.e. labels), percentage of each label.
    """
    ## Initialization
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xmls = glob.glob("{:s}/Annotations/*.xml".format(data_dir))
    #Logger.basic("Number of validation images found: {:d}".format(len(xmls)))
    model = DNN(model, device="cuda", score_threshold=confidence_threshold)
    images_bf = {}
    if subtract_brightfield:
        print("Note: Detection will be performed on brightfield subtracted images.")
        _data_dir = Path(__file__).resolve().parents[1].joinpath("data")
        _img_path = _data_dir.joinpath("brightfield_624_640.tif")
        if not _img_path.exists():
            print(f"Missing brightfield image for the 45kfps lens! {str(_img_path)}")
        else:
            _img_temp = cv.imread(str(_img_path), cv.IMREAD_GRAYSCALE)
            _key = "_".join([str(_i) for _i in _img_temp.shape[0:2]])
            images_bf[_key] = _img_temp

        _img_path = _data_dir.joinpath("brightfield_384_384.tif")
        if not _img_path.exists():
            print(f"Missing brightfield image for the 90kfps lens! {str(_img_path)}")
        else:
            _img_temp = cv.imread(str(_img_path), cv.IMREAD_GRAYSCALE)
            _key = "_".join([str(_i) for _i in _img_temp.shape[0:2]])
            images_bf[_key] = _img_temp
        
        if len(images_bf) == 0:
            print("No brightfield images loaded when ask to subtract brightfield image!")
            exit(1)

    # Run benchmark
    stats = {"particle_no-bubble_circle" : 0, 
             "particle_no-bubble_non-circle" : 0,
             "particle_bubble_circle" : 0,
             "particle_bubble_non-circle" : 0,
             "shell_circle" : 0,
             "shell_non-circle" : 0,
             "agglomerate" : 0}

    shape_accuracy = {"particle_circle" : 0, "particle_non-circle": 0,
                      "shell_circle" : 0, "shell_non-circle": 0, "agglomerate_non-circle" : 0}
    
    #labelled_data = {}
    total_num_ptcls = 0
    total_conf_matrix = {"true_positive": 0, "false_positive": 0, "false_negative": 0}

    # Store everything in a nested dict, as the benchmark script of GMM does.
    dict_conf_matrix = {}
    labeled_bbox = {}
    labeled_images = {}
    labeled_particles = {}
    for xml in xmls:
        # Aera threshold is already applied.
        ps, file_name = parse_pascal_xml(xml, area_threshold=area_threshold) # Labelled particle from this single file.
        re_obj = re.match(IMAGE_FILE_PATTERN, file_name)
        video_id = re_obj.group(1)
        image_id = re_obj.group(3)
        img_path = str(Path(data_dir).joinpath(f"images/{file_name}"))
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        if video_id not in labeled_bbox:
            labeled_bbox[video_id] = {}
            labeled_images[video_id] = {}
            labeled_particles[video_id] = {}
            dict_conf_matrix[video_id] = {}
        labeled_bbox[video_id][image_id] = [p.get_bbox_torch() for p in ps]
        labeled_images[video_id][image_id] = img
        labeled_particles[video_id][image_id] = ps

    for video_id in labeled_particles.keys():
        video_conf_matrix = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for image_id in labeled_particles[video_id].keys():
            img = labeled_images[video_id][image_id]
            img_orig = np.copy(img)
            ps = labeled_particles[video_id][image_id]
            gt_bbox = labeled_bbox[video_id][image_id]

            # ----------------------------------- Benchmark ------------------------------------#
            # Shape accuracy
            total_num_ptcls += len(ps)
            for p in ps:
                stats[p.get_label()] += 1
                shape = detect_shape(p, img)
                if shape == p.get_shape():
                    shape_accuracy["{:s}_{:s}".format(p.get_type(), p.get_shape())] += 1
        
            # Localization Accuracy
            if subtract_brightfield:
                _key = "_".join([str(_i) for _i in img.shape[0:2]])
                if _key not in images_bf:
                    print(f"No brightfield image exist for the resolution of this labelled image: {_key}")
                    exit(1)
                img_bf = images_bf[_key]
                # Realign peak after subtraction to mimic the background the style image.
                img, *_ = sbf_shifting(img, img_bf, scale=1, shift_back=True)
            pred_bbox, mask = model.predict(img)
            pred_bbox = np.round(pred_bbox).astype(np.int64)
            #Logger.detail("Number of detected particles in {:s}: {:d}".format(Path(img_path).name, len(pred_bbox)))
            seen_gt, seen_pred = compare_bbox(gt_bbox, pred_bbox, allow_enclose=False)
        
            # Draw the bbox comparison results out.
            video_dir = outdir.joinpath(f"video_{video_id}")
            video_dir.mkdir(parents = True, exist_ok=True)
            compare_dir = video_dir.joinpath("bbox_comparison")
            compare_dir.mkdir(exist_ok=True)
            
            img_white = draw_comparison(gt_bbox, seen_gt, pred_bbox, seen_pred, shape=(img.shape[0], img.shape[1]))
            cv.imwrite(str(compare_dir.joinpath(f"white_{image_id}.png")), img_white)
            
            img_white_padded = draw_comparison(gt_bbox, seen_gt, pred_bbox, seen_pred,
                                                padding=30, shape=(img.shape[0], img.shape[1]))
            cv.imwrite(str(compare_dir.joinpath(f"white_padded_{image_id}.png")), img_white_padded)

            img_white_padded_id = draw_comparison(gt_bbox, seen_gt, pred_bbox, seen_pred, id=True,
                                                    padding=30, shape=(img.shape[0], img.shape[1]))
            cv.imwrite(str(compare_dir.joinpath(f"white_padded_id_{image_id}.png")), img_white_padded_id)

            # Draw on the original image
            img_out = np.copy(img_orig)
            img_out = draw_comparison(gt_bbox, seen_gt, pred_bbox, seen_pred, img=img_out,
                                    shape=(img.shape[0], img.shape[1]))
            cv.imwrite(str(compare_dir.joinpath(f"orig_{image_id}.png")), img_out)

            conf_matrix = get_confusion_matrix(seen_gt, seen_pred) # Dict of True Positive, True Negative
            #print(video_id, image_id, conf_matrix)
            video_conf_matrix = merge_confusion_matrix(video_conf_matrix, conf_matrix)
            dict_conf_matrix[video_id][image_id] = conf_matrix

        total_conf_matrix = merge_confusion_matrix(total_conf_matrix, video_conf_matrix)
        write_accuracy(str(video_dir.joinpath("accuracy.txt")), video_conf_matrix)
        with open(str(video_dir.joinpath("confusion_matrix.txt")), "w") as f:
            json.dump(dict_conf_matrix[video_id], f)

    # Output for all labelled data for videos and images.
    num_particle = stats["particle_bubble_circle"] + stats["particle_bubble_non-circle"] + \
                stats["particle_no-bubble_circle"] + stats["particle_no-bubble_non-circle"]
    num_particle_circle = stats["particle_bubble_circle"] + stats["particle_no-bubble_circle"]
    num_particle_non_circle = stats["particle_bubble_non-circle"] + stats["particle_no-bubble_non-circle"]
    num_shell = stats["shell_circle"] + stats["shell_non-circle"]
    num_agglomerate = stats["agglomerate"]

    write_accuracy(str(outdir.joinpath("accuracy.txt")), total_conf_matrix)
    with open(str(outdir.joinpath("confusion_matrix.txt")), "w") as f:
        json.dump(dict_conf_matrix, f)
    
    with open(outdir.joinpath("shape_accuracy.txt"), "w") as f:
        f.write("Total number of labelled images = {:d}\n".format(len(xmls)))
        f.write("Total number of labelled particles = {:d}\n".format(total_num_ptcls))
        for l in stats:
            f.write("Number and percentages of label {:s} = {:3d} {:.3f}%\n".format(l, stats[l],
                    100*float(stats[l])/total_num_ptcls))

        # Shape accuracies:
        f.write("\n")
        f.write("Shape accuracy {:s} = {:.3f}%\n".format("particle_circle",
                100 * float(shape_accuracy["particle_circle"]) / num_particle_circle))
        if num_particle_non_circle > 0:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("particle_non-circle",
                    100 * float(shape_accuracy["particle_non-circle"]) / num_particle_non_circle))
        else:
            f.write("Shape accuracy {:s} : -1\n".format("particle_non-circle"))
        
        if stats["shell_circle"] > 0:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("shell_circle",
                    100 * float(shape_accuracy["shell_circle"]) / stats["shell_circle"]))
        else:
            f.write("Shape accuracy {:s} = -1\n".format("shell_circle"))
        
        if stats["shell_non-circle"] > 0:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("shell_non-circle",
                    100 * float(shape_accuracy["shell_non-circle"]) / stats["shell_non-circle"]))
        else:
            f.write("Shape accuracy {:s} = -1\n".format("shell_non-circle"))

        if stats["agglomerate"] > 0:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("agglomerate as non-circle",
                    100 * float(shape_accuracy["agglomerate_non-circle"]) / stats["agglomerate"]))
        else:
            f.write("Shape accuracy {:s} = -1\n".format("agglomerate as non-circle"))

    
# ------------------------------------ Helper function ----------------------------------------- #
def write_accuracy(file, conf_matrix):
    with open(file, "w") as f:
        prec = calc_precision(conf_matrix)
        recall = calc_recall(conf_matrix)
        for key in conf_matrix:
            f.write(f"{key} {conf_matrix[key]}\n")
        f.write(f"Precision {prec}\n")
        f.write(f"Recall {recall}\n")


if __name__ == "__main__":
    benchmark()