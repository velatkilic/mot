import unittest
from src.datagen.bead_gen import Beads


class TestBeads(unittest.TestCase):
    def setUp(self):
        self.bead = Beads()

    def test_gen_sample(self):
        img, seg, bbox = self.bead.genSample()
        self.assertIsNotNone(img)
        self.assertIsNotNone(seg)
        self.assertIsNotNone(bbox)