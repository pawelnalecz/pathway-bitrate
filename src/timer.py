import time
import numpy as np


class Timer():

    def __init__(self, *, prefix=''):
        self._prefix = prefix

    def start(self):
        self._start_time = time.time()
        self._mid_points = [self._start_time]
        self._texts = []
    
    def log(self, text=''):
        self._mid_points.append(time.time())
        self._texts.append(text)
    
    def report(self):
        ellapsed = np.diff(self._mid_points)
        for text, el in zip(self._texts, ellapsed):
            print(f"{self._prefix}{text} took {el:.3f} s")
        if len(ellapsed):
            print(f"{self._prefix}Total time: {self._mid_points[-1] - self._start_time:.3f} s.")
        else:
            print(f"{self._prefix}Timer: No entries logged.")
