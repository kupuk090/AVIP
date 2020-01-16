from pathlib import Path

from PIL import Image, ImageFont, ImageDraw
import numpy as np

import csv

from core import LabImage


def create_symbol_images(symbol_list: list or str, img_size=(50, 50), font='TNR.ttf', font_size=52) -> None:
    for sym in symbol_list:
        im = Image.new('L', img_size, color='white')
        d = ImageDraw.Draw(im)
        f = ImageFont.truetype(font, font_size)
        mw, mh = img_size
        w, h = d.textsize(sym, font=f)
        d.text((((mw - w) // 2), (mh - h) // 2), sym, font=f)

        im_matr = np.array(im)
        mask = im_matr == 255
        rows = np.flatnonzero(np.sum(~mask, axis=1))
        cols = np.flatnonzero(np.sum(~mask, axis=0))

        crop = im_matr[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
        im = Image.fromarray(crop, 'L')

        im.save(sym + '.bmp')


def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


class Lab4(LabImage):
    def __init__(self, path=None, image=None):
        self.grayscale_matrix = None
        self.bin_matrix = None

        if len(np.array(image).shape) > 2:
            im_matr = np.where((np.sum(np.array(image), axis=2) // 3) < 50, 0, 255)
        else:
            im_matr = np.where(np.array(image) < 50, 0, 255)
        mask = im_matr == 255
        rows = np.flatnonzero(np.sum(~mask, axis=1))
        cols = np.flatnonzero(np.sum(~mask, axis=0))

        crop = im_matr[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
        im = Image.fromarray(np.uint8(crop), 'L')

        if path is not None:
            super(Lab4, self).__init__(path=Path(path).resolve())
        elif image is not None:
            super(Lab4, self).__init__(image=im)

        # print('HERE')

    def to_grayscale(self):
        if len(self.rgb_matrix.shape) > 2:
            gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3
        else:
            gray_matrix = self.rgb_matrix

        self.grayscale_matrix = gray_matrix

    def to_binary_image(self, threshold: int):
        if self.grayscale_matrix is not None:
            self.bin_matrix = np.where(self.grayscale_matrix < threshold, 0, 255)
        else:
            self.to_grayscale()
            self.to_binary_image(threshold)

    def calc_characteristics(self) -> dict:
        if self.bin_matrix is None:
            self.to_binary_image(50)

        m, n = self.bin_matrix.shape
        inv_bin_matr = np.where(self.bin_matrix == 255, 0, 1)

        weight = np.sum(inv_bin_matr)
        norm_weight = weight / (self.height * self.width)

        x_center = np.sum([x * f for (y, x), f in np.ndenumerate(inv_bin_matr)]) // weight
        y_center = np.sum([y * f for (y, x), f in np.ndenumerate(inv_bin_matr)]) // weight

        norm_x_center = (x_center - 1) / (m - 1)
        norm_y_center = (y_center - 1) / (n - 1)

        x_moment = np.sum([f * (y - y_center) ** 2 for (y, x), f in np.ndenumerate(inv_bin_matr)])
        y_moment = np.sum([f * (x - y_center) ** 2 for (y, x), f in np.ndenumerate(inv_bin_matr)])
        xy_45_moment = np.sum([f * (y - y_center - x + x_center) ** 2 for (y, x), f in np.ndenumerate(inv_bin_matr)]) // 2
        xy_135_moment = np.sum([f * (y - y_center + x - y_center) ** 2 for (y, x), f in np.ndenumerate(inv_bin_matr)]) // 2

        norm_x_moment = x_moment / (weight ** 2)
        norm_y_moment = y_moment / (weight ** 2)
        norm_xy_45_moment = xy_45_moment / (weight ** 2)
        norm_xy_135_moment = xy_135_moment / (weight ** 2)

        return {'weight': weight, 'norm_weight': norm_weight,
                'center': (x_center, y_center),
                'norm_center': (norm_x_center, norm_y_center),
                'moment': (x_moment, y_moment, xy_45_moment, xy_135_moment),
                'norm_moment': (norm_x_moment, norm_y_moment, norm_xy_45_moment, norm_xy_135_moment)}


# result = [('symbol',
#            'weight', 'norm_weight',
#            'x', 'y',
#            'norm_x', 'norm_y',
#            'hor_ax_moment', 'ver_ax_moment',
#            'norm_hor_ax_moment', 'norm_ver_ax_moment')]
# create_symbol_images("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
# for sym in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ":
#     im = Lab4(sym + '.bmp')
#     characteristics = im.calc_characteristics()
#     result.append((sym, characteristics['weight'], characteristics['norm_weight'], ) + characteristics['center'] +
#                   characteristics['norm_center'] + characteristics['moment'][:2] + characteristics['norm_moment'][:2])
#
# csv_writer(result, 'result.csv')
