#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2018 Ina (David Doukhan - http://www.ina.fr/)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
import random
import sys
import time
from typing import NamedTuple, Literal, Optional

import keras
import numpy as np
from pyannote.algorithms.utils.viterbi import viterbi_decoding
from skimage.util import view_as_windows as vaw
from tensorflow.keras.utils import get_file

from .features import media2feats
from .viterbi_utils import pred2logemission, diag_trans_exp, log_trans_exp


def _energy_activity(loge, ratio):
    threshold = np.mean(loge[np.isfinite(loge)]) + np.log(ratio)
    raw_activity = (loge > threshold)
    return viterbi_decoding(pred2logemission(raw_activity),
                            log_trans_exp(150, cost0=-5))


def _get_patches(mspec, w, step):
    h = mspec.shape[1]
    data = vaw(mspec, (w, h), step=step)
    data.shape = (len(data), w * h)
    data = (data - np.mean(data, axis=1).reshape((len(data), 1))) / np.std(data, axis=1).reshape((len(data), 1))
    lfill = [data[0, :].reshape(1, h * w)] * (w // (2 * step))
    rfill = [data[-1, :].reshape(1, h * w)] * (w // (2 * step) - 1 + len(mspec) % 2)
    data = np.vstack(lfill + [data] + rfill)
    finite = np.all(np.isfinite(data), axis=1)
    data.shape = (len(data), w, h)
    return data, finite


def _binidx2seglist(binidx):
    """
    ss._binidx2seglist((['f'] * 5) + (['bbb'] * 10) + ['v'] * 5)
    Out: [('f', 0, 5), ('bbb', 5, 15), ('v', 15, 20)]
    
    #TODO: is there a pandas alternative??
    """
    curlabel = None
    bseg = -1
    ret = []
    for i, e in enumerate(binidx):
        if e != curlabel:
            if curlabel is not None:
                ret.append((curlabel, bseg, i))
            curlabel = e
            bseg = i
    ret.append((curlabel, bseg, i + 1))
    return ret


class DnnSegmenter:
    """
    DnnSegmenter is an abstract class allowing to perform Dnn-based
    segmentation using Keras serialized models using 24 mel spectrogram
    features obtained with SIDEKIT framework.

    Child classes MUST define the following class attributes:
    * nmel: the number of mel bands to used (max: 24)
    * viterbi_arg: the argument to be used with viterbi post-processing
    * model_fname: the filename of the serialized keras model to be used
        the model should be stored in the current directory
    * inlabel: only segments with label name inlabel will be analyzed.
        other labels will stay unchanged
    * outlabels: the labels associated the output of neural network models
    """

    def __init__(self, batch_size):
        # load the DNN model
        url = 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/models/'
        model_path = get_file(self.model_fname, url + self.model_fname, cache_subdir='inaSpeechSegmenter')
        self.nn = keras.models.load_model(model_path, compile=False)
        self.batch_size = batch_size

    def __call__(self, mspec, lseg, difflen=0):
        """
        *** input
        * mspec: mel spectrogram
        * lseg: list of tuples (label, start, stop) corresponding to previous segmentations
        * difflen: 0 if the original length of the mel spectrogram is >= 68
                otherwise it is set to 68 - length(mspec)
        *** output
        a list of adjacent tuples (label, start, stop)
        """

        if self.nmel < 24:
            mspec = mspec[:, :self.nmel].copy()

        patches, finite = _get_patches(mspec, 68, 2)
        if difflen > 0:
            patches = patches[:-int(difflen / 2), :, :]
            finite = finite[:-int(difflen / 2)]

        assert len(finite) == len(patches), (len(patches), len(finite))

        batch = []
        for lab, start, stop in lseg:
            if lab == self.inlabel:
                batch.append(patches[start:stop, :])
        # print(lseg)
        if len(batch) > 0:
            batch = np.concatenate(batch)
            rawpred = self.nn.predict(batch, batch_size=self.batch_size)
        nn_pred = rawpred if self.ret_nn_pred else None
        ret = []
        for lab, start, stop in lseg:
            if lab != self.inlabel:
                ret.append((lab, start, stop))
                continue

            l = stop - start
            r = rawpred[:l]
            rawpred = rawpred[l:]
            r[finite[start:stop] == False, :] = 0.5
            pred = viterbi_decoding(np.log(r), diag_trans_exp(self.viterbi_arg, len(self.outlabels)))
            for lab2, start2, stop2 in _binidx2seglist(pred):
                ret.append((self.outlabels[int(lab2)], start2 + start, stop2 + start))
        return ret, nn_pred


class SpeechMusic(DnnSegmenter):
    # Voice activity detection: requires energetic activity detection
    outlabels = ('speech', 'music')
    model_fname = 'keras_speech_music_cnn.hdf5'
    inlabel = 'energy'
    nmel = 21
    viterbi_arg = 150


class SpeechMusicNoise(DnnSegmenter):
    # Voice activity detection: requires energetic activity detection
    outlabels = ('speech', 'music', 'noise')
    model_fname = 'keras_speech_music_noise_cnn.hdf5'
    inlabel = 'energy'
    nmel = 21
    viterbi_arg = 80
    ret_nn_pred = False


class Gender(DnnSegmenter):
    # Gender Segmentation, requires voice activity detection
    outlabels = ('female', 'male')
    model_fname = 'keras_male_female_cnn.hdf5'
    inlabel = 'speech'
    nmel = 24
    viterbi_arg = 80
    ret_nn_pred = True


class ResultFrame(NamedTuple):
    gender: str
    start: float
    end: float


class Result(NamedTuple):
    frames: list[ResultFrame]
    file: str


class BatchResults(NamedTuple):
    results: list[Result]
    time_full: float
    time_avg: float
    successes: int
    messages: list[tuple[str, int]]


VadEngine = Literal['smn', 'sm']


class Segmenter:
    def __init__(self, vad_engine: VadEngine = 'smn', detect_gender: bool = True, batch_size: int = 32,
                 energy_ratio: float = 0.03):
        """
        Load neural network models
        
        Input:

        'vad_engine' can be 'sm' (speech/music) or 'smn' (speech/music/noise)
                'sm' was used in the results presented in ICASSP 2017 paper
                        and in MIREX 2018 challenge submission
                'smn' has been implemented more recently and has not been evaluated in papers
        
        'detect_gender': if False, speech excerpts are return labelled as 'speech'
                if True, speech excerpts are splitted into 'male' and 'female' segments

        'batch_size' : large values of batch_size (ex: 1024) allow faster processing times.
                They also require more memory on the GPU.
                default value (32) is slow, but works on any hardware
        """
        # set energy ratio for 1st VAD
        self.energy_ratio = energy_ratio

        # select speech/music or speech/music/noise voice activity detection engine
        assert vad_engine in ['sm', 'smn']
        if vad_engine == 'sm':
            self.vad = SpeechMusic(batch_size)
        elif vad_engine == 'smn':
            self.vad = SpeechMusicNoise(batch_size)

        # load gender detection NN if required
        assert detect_gender in [True, False]
        self.detect_gender = detect_gender
        if detect_gender:
            self.gender = Gender(batch_size)

    def segment_feats(self, mspec, loge, difflen, start_sec):
        """
        do segmentation
        require input corresponding to wav file sampled at 16000Hz
        with a single channel
        """

        # perform energy-based activity detection
        lseg = []
        for lab, start, stop in _binidx2seglist(_energy_activity(loge, self.energy_ratio)[::2]):
            if lab == 0:
                lab = 'noEnergy'
            else:
                lab = 'energy'
            lseg.append((lab, start, stop))

        # perform voice activity detection
        lseg = self.vad(mspec, lseg, difflen)[0]

        # perform gender segmentation on speech segments
        if self.detect_gender:
            lseg, ret_nn = self.gender(mspec, lseg, difflen)
            return [(lab, ret_nn[1]) for lab, start, stop in lseg]
            # print(ret_nn.shape)

        return [(lab, start_sec + start * .02, start_sec + stop * .02) for lab, start, stop in lseg]

    def __call__(self, filename: os.PathLike, start_sec: int = 0, stop_sec: Optional[int] = None):
        """
        Return segmentation of a given file
                * convert file to wav 16k mono with ffmpeg
                * call NN segmentation procedures
        * filename: path to the media to be processed (including remote url)
                may include any format supported by ffmpeg
        * start_sec (seconds): sound stream before start_sec won't be processed
        * stop_sec (seconds): sound stream after stop_sec won't be processed
        """
        mspec, loge, difflen = media2feats(filename, start_sec, stop_sec)
        return self.segment_feats(mspec, loge, difflen, start_sec)
