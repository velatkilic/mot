from xmot.datagen.bead_gen import bead_data_to_file
from xmot.datagen.bead_gen import Beads
from xmot.datagen.style_data_gen import StyleDatasetGen
from xmot.dataset import Dataset
import torch
import torchvision.models as models
import numpy as np
import os
import click

@click.command()
@click.argument("style_dir")
@click.argument("num", type=int)
@click.argument("outdir")
@click.option("--model", type=str, default=None, help="Path to pre-downloaded pyTorch model.")
@click.option("--max-radius", default=40, type=int)
@click.option("--min-radius", default=3, type=int)
@click.option("--max-beads", default=10, type=int)
@click.option("--min-beads", default=2, type=int)
@click.option("--img-size", default=384, type=int, help="Dimension of generated images. "\
              "The size of content images cannot be larger than style images. Therefore, default is "\
              "set to 384.")
@click.option("--sigma", default=1, help="Standard deviation of gaussian filter. Negative value to shut it off.")
@click.option("--save-orig", is_flag=True, default=False, help="Save the original content image before style transfer.")
@click.option("--allow-overlap", is_flag=True, help="Allow overlap or not.")
#@click.option("--debug", is_flag=True, default=True)
def generate(style_dir, num, outdir, model, max_radius, min_radius, max_beads, min_beads,
             img_size, sigma, save_orig, allow_overlap):
    #if debug:
    no_overlap = not allow_overlap # Default is True, i.e. don't allow overlap.

    print("Setting:")
    print("Model path:", model)
    print("max-radius", max_radius)
    print("min-radius", min_radius)
    print("max-beads", max_beads)
    print("min-beads", min_beads)
    print("img-size", img_size)
    print("sigma", sigma)
    print("save-orig", save_orig)
    print("no-overlap", no_overlap)

    os.makedirs(outdir, exist_ok=True)
    style_images = Dataset(image_folder = style_dir)
    np.random.seed(0) # Make the training data reproducible.
    
    bead_generator = Beads(side=img_size, beadradMax=max_radius, beadradMin=min_radius,
        numbeadsMax=max_beads, numbeadsMin=min_beads, sigma=sigma, no_overlap=no_overlap)
    style_data_gen = StyleDatasetGen(bead_generator, model=model, style_imgs=style_images, outFolder=outdir,
                                    N=num, save_orig=save_orig)
    style_data_gen.gen_dataset()

if __name__ == "__main__":
    generate()
