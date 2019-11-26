from pathlib import Path

from PIL import Image
import numpy as np

from core import LabImage


class Lab3(LabImage):
    def __init__(self, path=None):
        self.grayscale_matrix = None
        self.gradient_matrix = None

        super(Lab3, self).__init__(path=Path(path).resolve())

    def to_grayscale(self):
        gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3

        self.grayscale_matrix = gray_matrix

    def scharr_operator(self, threshold=None):
        gx = np.array([[3, 10, 3],
                       [0, 0, 0],
                       [-3, -10, -3]])
        gy = np.array([[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]])

        if self.grayscale_matrix is None:
            self.to_grayscale()

        gradient_matrix = np.empty(self.grayscale_matrix.shape)
        for (x, y), _ in np.ndenumerate(self.grayscale_matrix[1: -1, 1: -1]):
            a = self.grayscale_matrix[x: x + 3, y: y + 3]
            gradient_matrix[x + 1, y + 1] = np.sqrt(np.sum(gx * a) ** 2 + np.sum(gy * a) ** 2)
        self.gradient_matrix = gradient_matrix * 255 / np.max(gradient_matrix)

        if threshold is None:
            self.result = Image.fromarray(np.uint8(self.gradient_matrix), 'L')
        else:
            gradient_matrix = np.where(self.gradient_matrix < threshold, 0, 255)
            self.result = Image.fromarray(np.uint8(gradient_matrix), 'L')


im = Lab3("sample_2.bmp")
im.scharr_operator()
im.show()
