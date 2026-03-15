#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from layers import Layer

class ActivationLayer(Layer):

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0
    
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        ## COMPLETE
        sigmoid = self.activation_function(input)
        return sigmoid * (1 - sigmoid)  


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        ## COMPLETE
        return np.where(input > 0, 1, 0)
    
class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input):
        shifted = input - np.max(input, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def derivative(self, input):
        # não vamos usar esta derivada diretamente
        # quando combinamos softmax + categorical cross entropy
        return np.ones_like(input)

    def backward_propagation(self, output_error):
        # com categorical cross entropy, o gradiente já vem simplificado
        return output_error
    
