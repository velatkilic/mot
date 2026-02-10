#!/usr/bin/python3
import click
import cv2 as cv

@click.command()
@click.argument("video")
@click.argument("output_dir")
@click.option("--prefix", default=None, type=str, help="Prefix before frame index for naming the output images.")
def extract(video, output_dir, prefix):
    """
    Extract frames of VIDEO and write them into OUTPUT_DIR.
    """
    video_cap = cv.VideoCapture(video)
    counter = 0
    while True:
        ret, img = video_cap.read()
        if not ret: break
        if prefix != None:
            cv.imwrite("{:s}/{:s}_{:d}.png".format(output_dir, prefix, counter), img)
        else:
            cv.imwrite("{:s}/{:s}_{:d}.png".format(output_dir, video.split("/")[-1].split(".")[0], counter), img)
        counter = counter + 1

if __name__ == "__main__":
    extract()