from pathlib import Path
import cv2 as cv
import os
import click
from xmot.digraph.utils import contrast_stretch


@click.command()
@click.argument("input")
@click.argument("output")
@click.option("--contrast", default = 0.1, help="Percentage of pixels being saturated.")
def contrast(input, output, contrast):
    """
    Contrast INPUT image with CONTRAST percentage of pixels being saturated and write to OUTPUT.
    """
    img_orig = cv.imread(input)
    img_stretched = contrast_stretch(img_orig, saturation = contrast)
    cv.imwrite(output, img_stretched)


if __name__ == "__main__":
    contrast()