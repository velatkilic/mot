from typing import List, Tuple
import pandas as pd
import xml.etree.ElementTree as ET

from xmot.logger import Logger
from xmot.digraph.particle import Particle

"""
Parser of xmot.digraph.particle in different formats.
"""

def load_blobs_from_excel(file_name: str) -> List[Particle]:
    """ A temporary io function to load data from Kerri-Lee's excel data.

    The function assumes a specific format of the excel data, and will be replaced by more 
    general format later.
    """
    data_id = pd.read_excel(file_name, sheet_name="Particle ID", engine="openpyxl")
    data_pos = pd.read_excel(file_name, sheet_name="Raw_data", engine="openpyxl")

    # remove N/A
    data_id = data_id.fillna(0).astype(int)

    particles = []
    for i in range(0, len(data_id)):
        row_id = data_id.loc[i]
        row_pos = data_pos.loc[i]

        row_id = row_id[row_id.gt(0)]
        if len(row_id) <= 1:
            # Has only "Act_frame" column and no ided particle in this frame.
            continue

        time_frame = row_id["Act_frame"]
        # Iterate all particles ided in this frame
        for j in range(1, len(row_id)):   
            id = row_id[j]
            pos = [row_pos[2 * id], row_pos[2 * id + 1]]
            # Has no bubble info and predicted positions for now.
            particles.append(Particle(id, time_frame, pos))
    
    return particles

def load_blobs_from_text(file_name: str) -> List[Particle]:
    particles = []
    with open(file_name, "r") as f:
        for line in f:
            terms = line.replace(" ", "").split(",")
            terms = [int(term) for term in terms]
            if len(terms) < 8:
                Logger.warning("Invalid blob info: {:s}".format(line))
            else:
                position = [terms[0], terms[1]]   # x1, y1
                bbox = [terms[4], terms[5]]       # width, height
                idx = terms[6]
                time_frame = terms[7]
                particles.append(Particle(position, bbox=bbox, id=idx, time_frame=time_frame))
    return particles

def parse_pascal_xml(file: str):
    """
    Parse labelled images in PASCAL VOC format.

    Images are labelled by LabelImg with tags:
        particle_no-bubble_circle
        particle_no-bubble_non-circle
        particle_bubble_circle
        particle_bubble_non-circle
        shell_circle
        shell_non-circle
        agglomerate

    Return:
        List[Particle]: List of particles parsed from FILE.
        str: path to the image corresponding to FILE.
    """
    doc = ET.parse(file)
    img_path = doc.find("path").text
    particles = []
    for obj in doc.findall("object"): # Only find direct child of doc.
        properties = obj.find("name").text.split("_")
        p_type=properties[0]
        p_shape=""
        p_bubble=None
        if p_type == "particle":
            p_shape = properties[2]
            if properties[1] == "bubble":
                p_bubble = Particle([0,0], [1,1]) # Dummy object.
        elif p_type == "shell":
            p_shape = properties[1]
            p_bubble = Particle([0, 0], [1, 1]) # Dummy object. A shell must have bubble.
        elif p_type == "agglomerate":
            p_shape = "non-circle"
            p_bubble = None
        p_bbox = obj.find("bndbox")
        xmin = int(p_bbox.find("xmin").text)
        ymin = int(p_bbox.find("ymin").text)
        xmax = int(p_bbox.find("xmax").text)
        ymax = int(p_bbox.find("ymax").text)
        p = Particle(position=[xmin, ymin], bbox=[xmax - xmin, ymax - ymin],
                     type=p_type, shape=p_shape, bubble=p_bubble, path_img = img_path)
        particles.append(p)
    particles.sort(key=lambda p: p.get_position()[0]) # sort in ascending order of x.
    id = 0
    for p in particles:
        p.set_id(id)
        id += 1
    return particles, img_path
        

    
    