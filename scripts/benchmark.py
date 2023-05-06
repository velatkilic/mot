import click
import glob
import cv2 as cv
from operator import itemgetter
from pathlib import Path

from xmot.digraph.parser import parse_pascal_xml
from xmot.analyzer.shapeDetector import detect_shape
from xmot.digraph.particle import Particle
from xmot.mot.detectors import DNN
from xmot.logger import Logger
from xmot.config import AREA_THRESHOLD

@click.group()
def benchmark():
    pass

@benchmark.command()
@click.argument("data_dir")
@click.argument("model")
@click.argument("output")
@click.option("--debug-output", type=str, default="", help="Folder to write debug images.")
@click.option("--area-threshold", type=int, default=AREA_THRESHOLD)
@click.option("-t", "--tolerance", type=int, default=5, help="Tolerance in pixel when checking position equivalance. Inclusive.")
def batch(data_dir, model, output, area_threshold, tolerance, debug_output):
    """
    Collect all labelled data from .xml files in DATA_DIR and detect particles in corresponding
    images with MODEL. Then write bencharmk results to the OUTPUT file.

    \b
    Note:
    1. This script assumes that in SRC_DIR there exist images sharing same name as the .xml files
    but with .png suffix.
    2. In OUTPUT file, we will record statistics about the labelled data: total number of images,
    total number of particles (i.e. labels), percentage of each label.
    """
    xmls = glob.glob("{:s}/*.xml".format(data_dir))
    Logger.basic("Number of validation images found: {:d}".format(len(xmls)))
    model = DNN(model, device="cuda:0")
    stats = {"particle_no-bubble_circle" : 0, 
             "particle_no-bubble_non-circle" : 0,
             "particle_bubble_circle" : 0,
             "particle_bubble_non-circle" : 0,
             "shell_circle" : 0,
             "shell_non-circle" : 0,
             "agglomerate" : 0}

    shape_accuracy = {"particle_circle" : 0, "particle_non-circle": 0,
                      "shell_circle" : 0, "shell_non-circle": 0, "agglomerate_non-circle" : 0}
    detection_accuracy = {"true_positive" : 0, "false_positive" : 0, "false_negative": 0}
    
    labelled_data = {}
    total_num_ptcls = 0
    for xml in xmls:
        ps, file_name = parse_pascal_xml(xml, area_threshold=area_threshold) # Labelled particle from this single file.
        img_path = str(Path(data_dir).joinpath(file_name))
        labelled_data[img_path] = []
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        ps[:] = [p for p in ps if p.get_area() > area_threshold]
        total_num_ptcls += len(ps)
        for p in ps:
            #if p.get_area() < area_threshold:
            #    ps.remove(p)
            #    continue
            labelled_data[img_path].append(p)
            stats[p.get_label()] += 1
            shape = detect_shape(p, img)
            if shape == p.get_shape():
                shape_accuracy["{:s}_{:s}".format(p.get_type(), p.get_shape())] += 1
        
        bboxes, mask = model.predict(img)
        Logger.detail("Number of detected particles in {:s}: {:d}".format(Path(img_path).name, len(bboxes)))
        detected_ptcls = []
        for bbox in bboxes:
            bbox = list(map(round, bbox))
            p = Particle(position = [bbox[0], bbox[1]], bbox = [bbox[2] - bbox[0], bbox[3] - bbox[1]])
            if p.get_area() < area_threshold:
                continue
            detected_ptcls.append(p)
        detected_ptcls.sort(key=lambda p: p.get_position()[0])

        # Check detection accuracy
        img_debug = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
        for p1 in ps:
            found = False
            for p2 in detected_ptcls[:]: # Create a copy. Otherwise, it creates the classic editing-while-iterating bug.
                if is_equivalent(p1.get_bbox_torch(), p2.get_bbox_torch(), tolerance):
                    detection_accuracy["true_positive"] += 1 # Labelled and detected.
                    detected_ptcls.remove(p2)
                    found = True
                    break
            if not found:
                detection_accuracy["false_negative"] += 1 # Labelled, but not not detected
            
            # Draw bboxes out for debug purposes.
            if not found:
                b = p1.get_bbox_torch()
                cv.rectangle(img_debug, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255), thickness = 1) # Red
            else:
                b = p2.get_bbox_torch()
                cv.rectangle(img_debug, (b[0], b[1]), (b[2], b[3]), color=(0, 255, 0), thickness = 1) # Green.

        detection_accuracy["false_positive"] += len(detected_ptcls) # Not labelled, but detected.
        for p in detected_ptcls:
            b = p.get_bbox_torch()
            cv.rectangle(img_debug, (b[0], b[1]), (b[2], b[3]), color=(255, 0, 0), thickness = 1) # Blue
        if debug_output != "":
            debug_img_path = Path(debug_output).joinpath(Path(img_path).name)
            cv.imwrite(str(debug_img_path), img_debug)

    num_particle = stats["particle_bubble_circle"] + stats["particle_bubble_non-circle"] + \
                   stats["particle_no-bubble_circle"] + stats["particle_no-bubble_non-circle"]
    num_particle_circle = stats["particle_bubble_circle"] + stats["particle_no-bubble_circle"]
    num_particle_non_circle = stats["particle_bubble_non-circle"] + stats["particle_no-bubble_non-circle"]
    num_shell = stats["shell_circle"] + stats["shell_non-circle"]
    num_agglomerate = stats["agglomerate"]

    with open(output, "w") as f:
        f.write("Total number of labelled images = {:d}\n".format(len(xmls)))
        f.write("Total number of particles = {:d}\n".format(total_num_ptcls))
        for l in stats:
            f.write("Percentage of label {:s} = {:.3f}%\n".format(l, 100*float(stats[l])/total_num_ptcls))

        # Shape accuracies:
        f.write("\n")
        f.write("Shape accuracy {:s} = {:.3f}%\n".format("particle_circle",
                100 * float(shape_accuracy["particle_circle"]) / num_particle_circle))
        if num_particle_non_circle > 0:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("particle_non-circle",
                    100 * float(shape_accuracy["particle_non-circle"]) / num_particle_non_circle))
        else:
            f.write("Shape accuracy {:s} : No particle.\n".format("particle_non-circle"))
        
        if stats["shell_circle"] > 0:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("shell_circle",
                    100 * float(shape_accuracy["shell_circle"]) / stats["shell_circle"]))
        else:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("shell_circle",
                    100 * float(shape_accuracy["shell_circle"]) / -1))
        
        if stats["shell_non-circle"] > 0:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("shell_non-circle",
                    100 * float(shape_accuracy["shell_non-circle"]) / stats["shell_non-circle"]))
        else:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("shell_circle",
                    100 * float(shape_accuracy["shell_circle"]) / -1))

        if stats["agglomerate"] > 0:
            f.write("Shape accuracy {:s} = {:.3f}%\n".format("agglomerate as non-circle",
                    100 * float(shape_accuracy["agglomerate_non-circle"]) / stats["agglomerate"]))
        else:
            f.write("Shape accuracy {:s} = {:s}%\n".format("agglomerate", "N/A"))

        # Detection accuracy:
        for key, value in detection_accuracy.items():
            f.write("Detection accuracy {:s} = {:.3f}%\n".format(key, 100 * float(value) / total_num_ptcls))
    
    # Benchmark results:
    #    Position: true positive, false positive, false negative;
    #    Shape: true, false;

#@benchmark.command()
#@click.argument("img")
#@click.argument("xml")
#@click.arugment("model")
#@click.argument("output")
#@click.option("--debug-output", type=str, default="", help="Folder to write debug images.")
#@click.option("--area-threshold", type=int, default=10)
#@click.option("-t", "--tolerance", type=int, default=5, help="Tolerance in pixel when checking position equivalance. Inclusive.")
#def single_image(img, xml, model, output, ):

def is_equivalent(bbox1, bbox2, tolerance):
    diff = list(map(lambda x, y: abs(x-y), bbox1, bbox2))
    if max(diff) <= tolerance:
        return True
    return False

if __name__ == "__main__":
    benchmark()