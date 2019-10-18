from PIL import Image
import numpy as np

from core import LabImage, timeit
from exceptions import WrongWindowSize


class Lab1(LabImage):
    def __init__(self, path=None):
        self.grayscale_matrix = None
        self.bin_matrix = None

        super(Lab1, self).__init__(path=path)

    def to_grayscale(self):
        gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3
        gray_im = Image.fromarray(np.uint8(gray_matrix), 'L')

        self.grayscale_matrix = gray_matrix
        self.result = gray_im

    @timeit
    def binarization(self, rsize=3, Rsize=15, eps=15):
        def otsu_global(matrix: np.ndarray):
            n_curr = 0
            T_res = 0
            M0_res = 0
            M1_res = 0

            p_tmp = np.unique(matrix, return_counts=True)
            p = p_tmp[1] / matrix.size

            for t in range(matrix.min(), matrix.max()):
                w0 = p[p_tmp[0] <= t].sum() if p[p_tmp[0] <= t].sum() > 0.00001 else 0.00001
                w1 = 1 - w0 if 1 - w0 > 0.00001 else 0.00001
                M0 = (p_tmp[0][p_tmp[0] <= t] * p[p_tmp[0] <= t]).sum() / w0
                M1 = (p_tmp[0][p_tmp[0] > t] * p[p_tmp[0] > t]).sum() / w1
                D0 = (p[p_tmp[0] <= t] * np.square(p_tmp[0][p_tmp[0] <= t] - M0)).sum()
                D1 = (p[p_tmp[0] > t] * np.square(p_tmp[0][p_tmp[0] > t] - M1)).sum()

                n = (w0 * w1 * (M0 - M1)**2) // (w0 * D0 + w1 * D1)
                if n >= n_curr:
                    n_curr = n
                    T_res = t
                    M0_res = M0
                    M1_res = M1

            return T_res, M0_res, M1_res

        # @timeit
        def split_submatrix(matrix: np.ndarray, submat1_shape: tuple, submat2_shape: tuple):
            p, q = submat1_shape
            P, Q = submat2_shape
            m, n = matrix.shape

            bias_p = (P - p) // 2
            bias_q = (Q - q) // 2
            for x in range(0, m, p):
                for y in range(0, n, q):
                    yield (
                              (
                                  (x, (x + p) if (x + p - m) < 0 else m),
                                  (y, (y + q) if (y + q - n) < 0 else n)
                              ),
                              (
                                  ((x - bias_p) if (x - bias_p) > 0 else 0, (x + P - bias_p) if (x + P - bias_p) < m else m),
                                  ((y - bias_q) if (y - bias_q) > 0 else 0, (y + Q - bias_q) if (y + Q - bias_q) < n else n)
                              )
                    )

        def binarization_processor(matrix_ind: tuple, epsilon=eps):
            matrix_k_ind, matrix_K_ind = matrix_ind
            matrix_k = self.grayscale_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1],
                                             matrix_k_ind[1][0]: matrix_k_ind[1][1]]
            matrix_K = self.grayscale_matrix[matrix_K_ind[0][0]: matrix_K_ind[0][1],
                                             matrix_K_ind[1][0]: matrix_K_ind[1][1]]
            T, M0, M1 = otsu_global(matrix_K)

            if abs(M1 - M0) >= epsilon:
                self.bin_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1], matrix_k_ind[1][0]: matrix_k_ind[1][1]] = \
                    np.where(matrix_k < T, 0, 255)
            else:
                k_mean = matrix_k.mean()
                new_T = (M0 + M1) / 2
                self.bin_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1], matrix_k_ind[1][0]: matrix_k_ind[1][1]]\
                    .fill(0 if k_mean <= new_T else 255)

        if (not (rsize % 2) and not (Rsize % 2)) or ((rsize % 2) and (Rsize % 2)):
            if self.grayscale_matrix is None:
                self.to_grayscale()
            self.bin_matrix = self.grayscale_matrix.astype(np.uint8)

            for x in split_submatrix(self.bin_matrix, (rsize, rsize), (Rsize, Rsize)):
                binarization_processor(x)

            self.result = Image.fromarray(self.bin_matrix, 'L')

        else:
            raise WrongWindowSize("Rsize={} and rsize={} must be even or odd both together".format(Rsize, rsize))


im = Lab1("sample_3.bmp")

im.to_grayscale()
# im.save("sample_5_gray.bmp")
im.binarization(rsize=3, Rsize=15)
im.show()

