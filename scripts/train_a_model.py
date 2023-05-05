from xmot.datagen.bead_gen import BeadDatasetFile, collate_fn
from torch.utils.data import DataLoader
from xmot.mot.detectors import DNN
import torch
import click


@click.command()
@click.argument("dir1")
@click.argument("size1", type=int)
@click.argument("model_output")
@click.option("--dir2", default=None, type=str, help="Second training data folder.")
@click.option("--size2", default=-1, type=int, help="Size of data from the second training data folder.")
@click.option("--batch-size", default=2, type=int, help="Mini-batch size.")
@click.option("--epoch", default=20, type=int, help="Number of epoches of training.")
def train_a_model(dir1, size1, model_output, dir2, size2, batch_size, epoch):
    """
    A script specially used for training models used to benchmark the effect of style transfered data
    and pure background data.
    """
    dataset = BeadDatasetFile(dir1, len=size1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                            num_workers=0, drop_last=False)
    #print("Number of style training data:", len(dataloader)) # Print the number of iterations.

    model = DNN(device="cuda", model=None)
    model.train(dataloader, epoch=epoch, print_interval=200)

    if size2 > 0 and dir2 is not None:
        dataset2 = BeadDatasetFile(dir2, len=size2)
        dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                 num_workers=0, drop_last=False)
        model.train(dataloader2)

    # The state_dict has the same size as directly saving the model. Just save the model.
    model.save_model(model_output)

if __name__ == "__main__":
    train_a_model()