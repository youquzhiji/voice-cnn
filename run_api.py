import pandas

from inaSpeechSegmenter import Segmenter

if __name__ == '__main__':
    print('Starting')
    seg = Segmenter()
    data = seg('VT 150hz baseline example.mp3')
    df = pandas.DataFrame(data, columns=['Prediction', 'Start', 'End', 'Confidence'])
    print(df)