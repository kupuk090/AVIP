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
    def __init__(self, path=None):
        self.grayscale_matrix = None
        self.bin_matrix = None

        super(Lab4, self).__init__(path=Path(path).resolve())

    def to_grayscale(self):
        gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3

        self.grayscale_matrix = gray_matrix

    def to_binary_image(self, threshold: int):
        if self.grayscale_matrix is not None:
            self.bin_matrix = np.where(self.grayscale_matrix < threshold, 0, 255)
        else:
            self.to_grayscale()
            self.to_binary_image(threshold)

    def calc_characteristics(self) -> tuple:
        if self.bin_matrix is None:
            self.to_binary_image(50)

        m, n = self.bin_matrix.shape

        weight = np.sum(self.bin_matrix) // 255
        norm_weight = weight / (self.height * self.width)

        x_center = np.sum([x * f for (x, y), f in np.ndenumerate(self.bin_matrix)]) // (weight * 255)
        y_center = np.sum([y * f for (x, y), f in np.ndenumerate(self.bin_matrix)]) // (weight * 255)

        norm_x_center = (x_center - 1) / (m - 1)
        norm_y_center = (y_center - 1) / (n - 1)

        x_moment = np.sum([f * (x - x_center) ** 2 for (x, y), f in np.ndenumerate(self.bin_matrix)]) // 255
        y_moment = np.sum([f * (y - y_center) ** 2 for (x, y), f in np.ndenumerate(self.bin_matrix)]) // 255

        norm_x_moment = x_moment / (m ** 2 + n ** 2)
        norm_y_moment = y_moment / (m ** 2 + n ** 2)

        return (weight, norm_weight,
                x_center, y_center,
                norm_x_center, norm_y_center,
                x_moment, y_moment,
                norm_x_moment, norm_y_moment)


result = [('symbol',
           'weight', 'norm_weight',
           'x', 'y',
           'norm_x', 'norm_y',
           'hor_ax_moment', 'ver_ax_moment',
           'norm_hor_ax_moment', 'norm_ver_ax_moment')]
create_symbol_images("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
for sym in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ":
    im = Lab4(sym + '.bmp')
    result.append((sym,) + im.calc_characteristics())

csv_writer(result, 'result.csv')
