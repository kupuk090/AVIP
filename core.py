import os
import sys

from PIL import Image
import numpy as np

import time

from exceptions import ResultNotExist


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


class LabImage:
    def __init__(self, path=None):
        self.path = path
        self.result = None

        if path is not None:

            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(sys.path[0], path))

            self.orig = Image.open(path).convert("RGB")
            self.size = self.orig.size
            self.height, self.width = self.size
            self.rgb_matrix = np.array(self.orig)

    def read(self, path: str):
        self.path = path

        self.orig = Image.open(path).convert("RGB")
        self.size = self.orig.size
        self.height, self.width = self.size
        self.rgb_matrix = np.array(self.orig)

    def show(self):
        if self.result is None:
            self.orig.show()
        else:
            self.result.show()

    def save(self, name: str):
        if self.result is not None:
            self.result.save(name)
        else:
            raise ResultNotExist("No such results for saving it to {}".format(name))
