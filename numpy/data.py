#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Data:
    def __init__(self, X, y=None, features=None, label=None):
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        return self.X.shape

    def has_label(self):
        return self.y is not None