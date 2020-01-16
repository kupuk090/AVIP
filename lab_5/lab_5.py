from pathlib import Path

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd

import random

from core import LabImage
from lab_4.lab_4 import Lab4


def create_text_image(text: list or str, img_size=(50, 50), font='TNR.ttf', font_size=52) -> None:
    mw, mh = img_size
    mw *= len(text)
    mh = mh * len(text.split('\n')) * 2
    im = Image.new('L', (mw, mh), color='white')
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, font_size)
    w, h = d.textsize(text, font=f)
    d.text((((mw - w) // 2), (mh - h) // 2), text, font=f, spacing=20)

    im_matr = np.array(im)
    mask = im_matr == 255
    rows = np.flatnonzero(np.sum(~mask, axis=1))
    cols = np.flatnonzero(np.sum(~mask, axis=0))

    crop = im_matr[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
    im = Image.fromarray(crop, 'L')

    im.save('text.bmp')


class Lab5(LabImage):
    def __init__(self, path=None):
        self.grayscale_matrix = None
        self.bin_matrix = None

        self.letters_rect = []
        self.letters_characteristics = {}

        super(Lab5, self).__init__(path=Path(path).resolve())

        if getattr(self, 'bin_matrix', None) is None:
            self.to_binary_image(50)

        self.inv_bin_matr = np.where(self.bin_matrix == 255, 0, 1)

    def to_grayscale(self):
        gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3

        self.grayscale_matrix = gray_matrix

    def to_binary_image(self, threshold: int):
        if self.grayscale_matrix is not None:
            self.bin_matrix = np.where(self.grayscale_matrix < threshold, 0, 255)
        else:
            self.to_grayscale()
            self.to_binary_image(threshold)

    def get_ver_profile(self, matr=None):
        if matr is not None:
            return np.sum(matr, axis=1)
        else:
            return np.sum(self.inv_bin_matr, axis=1)

    def get_hor_profile(self, matr=None):
        if matr is not None:
            return np.sum(matr, axis=0)
        else:
            return np.sum(self.inv_bin_matr, axis=0)

    def get_zones(self, profile_matr: np.array, wsize=3):
        start = []
        end = []
        f = False
        for i in range(profile_matr.size - wsize + 1):
            if np.all(np.roll(profile_matr, -i)[0: wsize]) and not f:
                f = not f
                start.append(i)
            elif not np.any(np.roll(profile_matr, -i)[0: wsize]) and f:
                f = not f
                end.append(i)

        if len(start) > len(end):
            end.append(len(profile_matr))

        return [(start[i], end[i]) for i in range(len(start))]

    def text_segmentation(self):
        text_coords = []

        rows_coord = self.get_zones(self.get_ver_profile(), wsize=5)
        for i in rows_coord:
            s, e = i
            text_coords.append([((js, s), (je, e)) for (js, je) in self.get_zones(self.get_hor_profile(self.inv_bin_matr[s: e, ]), wsize=6)])

        letters_rect = []
        for ii in text_coords:
            tmp_letters_rect = []
            for i in ii:
                (sx, sy), (ex, ey) = i
                tmp_lst = [((sx + js, sy), (sx + je, ey)) for (js, je) in self.get_zones(self.get_hor_profile(self.inv_bin_matr[sy: ey, sx: ex]), wsize=1)]

                to_del_ind = []
                for j in range(1, len(tmp_lst)):
                    if ((tmp_lst[j-1][1][0] - tmp_lst[j-1][0][0]) / (tmp_lst[j][1][0] - tmp_lst[j][0][0])) > 1.85:
                        tmp_lst[j-1] = (tmp_lst[j-1][0], tmp_lst[j][1])
                        to_del_ind.append(j)

                if len(to_del_ind):
                    for j in sorted(to_del_ind, reverse=True):
                        tmp_lst.pop(j)
                tmp_letters_rect.append(tmp_lst)
            letters_rect.append(tmp_letters_rect)

        self.letters_rect = letters_rect

    def draw_rectangles(self):
        im = self.orig
        d = ImageDraw.Draw(im)
        for ii in self.letters_rect:
            for iii in ii:
                col = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in iii:
                    # col = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    d.line(((i[0][0], i[0][1]), (i[1][0], i[0][1])), fill=col, width=1)
                    d.line(((i[1][0], i[0][1]), (i[1][0], i[1][1])), fill=col, width=1)
                    d.line(((i[1][0], i[1][1]), (i[0][0], i[1][1])), fill=col, width=1)
                    d.line(((i[0][0], i[1][1]), (i[0][0], i[0][1])), fill=col, width=1)

        im.show()

    def text_recognizing(self):
        if getattr(self, 'letters_rect', None) is None:
            self.text_segmentation()

        for sym in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ":
            delta = 20
            f = ImageFont.truetype('TNR.ttf', 52)
            w, h = f.getsize(sym)
            w, h = map(lambda x: x + delta, (w, h))
            letter_im = Image.new('L', (w, h), color='white')
            d = ImageDraw.Draw(letter_im)
            d.text((delta // 2, delta // 2), sym, font=f)

            im_matr = np.array(letter_im)
            mask = im_matr == 255
            rows = np.flatnonzero(np.sum(~mask, axis=1))
            cols = np.flatnonzero(np.sum(~mask, axis=0))

            crop = im_matr[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
            letter_im = Image.fromarray(crop, 'L')

            self.letters_characteristics.update({sym: Lab4(image=letter_im).calc_characteristics()})

        text_list = []
        for ii in self.letters_rect:
            row_list = []
            for iii in ii:
                word_list = []
                for i in iii:
                    im = self.orig.crop(i[0] + i[1])
                    characteristics = Lab4(image=im).calc_characteristics()
                    df = pd.DataFrame(columns=['letter', 'distance'])

                    for k, v in self.letters_characteristics.items():
                        df = df.append({'letter': k,
                                        'distance': 1 - np.linalg.norm(np.array((v['norm_weight'],) +
                                                                                v['norm_center'] +
                                                                                v['norm_moment']) -
                                                                       np.array((characteristics['norm_weight'],) +
                                                                                characteristics['norm_center'] +
                                                                                characteristics['norm_moment']))},
                                       ignore_index=True)
                    df.sort_values(by=['distance'], ascending=False, inplace=True)
                    word_list.append(df)
                    print(df.head(1).letter.iloc[0], end='')
                row_list.append(word_list)
                print(' ', end='')
            text_list.append(row_list)
            print()

        to_file_list = []
        for ii in text_list:
            for iii in ii:
                for i in iii:
                    to_file_list.append(', '.join([f"({x}, {y})" for (x, y) in i.values]))
                    to_file_list.append('\n')
                to_file_list.append('\n')
            to_file_list.append('\n\n\n')

        with open('result.txt', "w", newline='') as file:
            for item in to_file_list:
                file.write(item)


create_text_image("Шалящий фавн прикинул объём горячих звезд этих вьюжных царств".upper() + '\n' +
                  "ЭТО ТЕСТОВЫЙ ТЕКСТ НА НОВОЙ СТРОКЕ" + '\n' +
                  "Съешь же ещё этих мягких французских булок да выпей чаю".upper(), font_size=52)
im = Lab5('text.bmp')
im.text_segmentation()
im.draw_rectangles()
im.text_recognizing()
