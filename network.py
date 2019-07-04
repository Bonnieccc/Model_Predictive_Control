#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer.links as L
import chainer.functions as F
from chainer import Chain, initializers
import numpy as np


class NN(Chain):
    def __init__(self, insize, outsize):
        super(NN, self).__init__(
            # bn1=L.BatchNormalization(insize),
            layer1=L.Linear(insize, 110, initialW=initializers.Normal(scale=0.05)),
            layer2=L.Linear(110, 100, initialW=initializers.Normal(scale=0.05)),
            layer3=L.Linear(100, outsize, initialW=np.zeros((outsize, 100), dtype=np.float32))
        )

    def __call__(self, x):
        # h = F.leaky_relu(self.bn1(x))
        h = F.leaky_relu(self.layer1(x))
        h = F.leaky_relu(self.layer2(h))
        h = self.layer3(h)

        return h
