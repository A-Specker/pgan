#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Florian Grimm, Vanessa Kirchner, Andreas Specker
# April 2018


import os
import sys
import argparse
import tensorflow as tf
import cv2
import numpy as np
from tensorpack import *
from tensorpack.utils.viz import stack_patches
import importlib


"""
To sample you need a trained model and the according Model.
    start train script or get one from
    /graphics/projects/scratch/student_datasets/cgpraktikum17/vat/flo_data/april/train_log


Example to sample:
    ./sample.py --res 8 --file 8x8.py --batch 16 --load model model-37500.data-00000-of-00001

Only works with stabilized models, to sample sample from transition adjust 'out_name'

args.file has to be in same folder as this script

"""



# Dont use this! Only if you dont want random noise
class ZData(DataFlow):
    def __init__(self, shape):
        super(ZData, self).__init__()
        self.shape = shape
        self.shape = shape
        self.eps = 0.0001
        self.ret = -1

# Gerneates random uniform noise for G to evaluate
class RandomZData(DataFlow):
    def __init__(self, shape):
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        while True:
            yield [np.random.uniform(0, 1, size=self.shape)]


# uses offline predictor sample generator output
def sample(model_path, output_name, BATCH_SIZE, Net, res):
    #output_name = 'gen/toRGBtemp_128_256/output'
    BATCH_SIZE = int(BATCH_SIZE)
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Net.Model(),
        input_names=['z'],
        output_names=[output_name, 'z'])
    inputNoise = RandomZData((BATCH_SIZE, 512))
    #inputNoise = ZData((BATCH_SIZE, 512))
    pred = SimpleDatasetPredictor(pred, inputNoise)
    its = 0
    for o in pred.get_result():
        o = o[0] + 1
        o = o * 128.0
        o = np.clip(o, 0, 255)
        o = o[:, :, :, ::-1]
        stack_patches(o, nr_row=10, nr_col=10, viz=True)

        # uncomment to safe the samples
        # for fc in o:
        #     its += 1
        #     name = '%i_%d' % (int(res), its,)
        #     name = name+ '.png'
        #     cv2.imwrite(name, fc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--res', help='Res of Imgs')
    parser.add_argument('--file', help='sample from which file')
    parser.add_argument('--batch', help='BatchSize used to train model')
    args = parser.parse_args()

    BATCH_SIZE = args.batch
    out_name = 'gen/toRGB_' + args.res + '_' + args.res + '/output'
    net_name = args.file.strip('.py')
    Net = importlib.import_module(net_name)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    sample(args.load, out_name, BATCH_SIZE, Net, args.res)
