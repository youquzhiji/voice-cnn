import time

import pandas

from inaSpeechSegmenter import Segmenter


class Timer:
    start: int

    def __init__(self):
        self.reset()

    def elapsed(self, reset: bool = True) -> float:
        t = (time.time_ns() - self.start) / 1000000
        if reset:
            self.reset()
        return t

    def log(self, *args):
        print(f'{self.elapsed():.0f}ms', *args)

    def reset(self):
        self.start = time.time_ns()


if __name__ == '__main__':
    print('Starting')
    seg = Segmenter()
    data = seg('VT 150hz baseline example.mp3')
    # data = seg('test.wav')
    df = pandas.DataFrame(data, columns=['Prediction', 'Start', 'End', 'Confidence'])
    print(df)

    timer = Timer()
    data = seg('VT 150hz baseline example.mp3')
    data = seg('VT 150hz baseline example.mp3')
    data = seg('VT 150hz baseline example.mp3')
    timer.log('Segmented x3')

