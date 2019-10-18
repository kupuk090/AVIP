from pathlib import Path

from PIL import Image
import numpy as np

from core import LabImage, timeit
from exceptions import WrongWindowSize, WrongRank


class Lab2(LabImage):
    def __init__(self, path=None):
        self.grayscale_matrix = None
        self.filtered_matrix = None

        super(Lab2, self).__init__(path=Path(path).resolve())

    def to_grayscale(self):
        gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3
        gray_im = Image.fromarray(np.uint8(gray_matrix), 'L')

        self.grayscale_matrix = gray_matrix

    def rank_filter(self, rank, wsize=3):
        def prepare_matrix(matrix: np.ndarray, window_size: int):
            bias = wsize // 2
            new_matrix = np.vstack((matrix[1:(bias + 1)][::-1], matrix, matrix[-(bias + 1):-1][::-1]))
            new_matrix = np.hstack((new_matrix[:, 1:(bias + 1)][:, ::-1], new_matrix, new_matrix[:, -(bias + 1):-1][:, ::-1]))

            return new_matrix

        if not wsize % 2:
            raise WrongWindowSize("wsize must be odd, positive and integer")

        if rank >= wsize**2 or rank < 0:
            raise WrongRank("rank must be positive and less than wsize*wsize")

        m, n = self.size

        if self.grayscale_matrix is None:
            self.to_grayscale()

        bias = wsize // 2
        prepared_matrix = prepare_matrix(self.grayscale_matrix, wsize)
        filtered_matrix = np.ndarray(self.grayscale_matrix.shape)
        for (x, y), _ in np.ndenumerate(self.grayscale_matrix):
            filtered_matrix[x, y] = sorted(prepared_matrix[x: x+wsize, y: y+wsize].flatten())[rank]

        self.filtered_matrix = np.uint8(filtered_matrix)
        self.result = Image.fromarray(self.filtered_matrix, 'L')


im = Lab2("sample_4.bmp")
im.rank_filter(6, wsize=3)
im.show()
