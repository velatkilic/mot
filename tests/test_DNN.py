from unittest import TestCase
import torch
from xmot.datagen.bead_gen import BeadDataset, Beads, collate_fn, BeadDatasetFile
from xmot.mot.detectors import DNN
import os


class TestDNN(TestCase):

    def test_DNN_train_with_beads(self):
        beads = Beads()
        dset = BeadDataset(beads, length=100)

        train_dataloader = torch.utils.data.DataLoader(dset, batch_size=2, shuffle=True,
                                                       collate_fn=collate_fn, num_workers=4)

        dnn = DNN()
        dnn.train(train_dataloader, epoch=1, print_interval=10)
        img, target = dset[0]
        bbox, mask = dnn.predict(img)

    def test_DNN_train_with_beads_from_file(self):
        filename = os.path.join(os.getcwd(), "train")
        dset = BeadDatasetFile(filename)

        train_dataloader = torch.utils.data.DataLoader(dset, batch_size=2, shuffle=True,
                                                       collate_fn=collate_fn, num_workers=4)

        dnn = DNN()
        dnn.train(train_dataloader, epoch=1, print_interval=10)
        img, target = dset[0]
        bbox, mask = dnn.predict(img)

    def test_DNN_train_with_style_beads_from_file(self):
        filename = os.path.join(os.getcwd(), "train_style")
        dset = BeadDatasetFile(filename)

        train_dataloader = torch.utils.data.DataLoader(dset, batch_size=2, shuffle=True,
                                                       collate_fn=collate_fn, num_workers=4)

        dnn = DNN()
        dnn.train(train_dataloader, epoch=1, print_interval=10)
        img, target = dset[0]
        bbox, mask = dnn.predict(img)


