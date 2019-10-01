from pathlib import Path

from PIL import Image
import numpy as np

from core import LabImage, timeit


class Lab2(LabImage):
    def __init__(self, path=None):
        super(Lab2, self).__init__(path=Path(path).resolve())

    def rank_filter(self, rank, wsize=3):
        pass
