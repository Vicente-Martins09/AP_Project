#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from losses import LossFunction, MeanSquaredError, CategoricalCrossEntropy
from optimizer import Optimizer
from metrics import accuracy, mse
import copy


class NeuralNetwork:
 
    def __init__(self, epochs = 100, batch_size = 128, optimizer = None,
                 learning_rate = 0.01, momentum = 0.90, verbose = False, 
                 loss = CategoricalCrossEntropy,
                 metric:callable = accuracy):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum= momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        # attributes
        self.layers = []    
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y = None,shuffle = True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            if y is not None:
                yield X[indices[start:end]], y[indices[start:end]]
            else:
                yield X[indices[start:end]], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        ## COMPLETE
        
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        ## for ...
        return error

    def fit(self, dataset, val_dataset = None, early_stopping = False, patience = 5):

        
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            raise ValueError("For multiclass classification, y should be one-hot encoded.")

        self.history = {}
        
        best_val_loss = np.inf
        best_layers = None
        epochs_without_improvement = 0
        
        for epoch in range(1, self.epochs + 1):
            # store mini-batch data for epoch loss and quality metrics calculation
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                # Forward propagation
                output = self.forward_propagation(X_batch, training=True)
                # Backward propagation
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # compute loss
            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
            else:
                metric = 'NA'
                
            val_loss = None
            val_metric = None
            
            if val_dataset is not None:
                val_output = self.forward_propagation(val_dataset.X, training=False)
                val_loss = self.loss.loss(val_dataset.y, val_output)
                val_metric = self.score(val_dataset, val_output)

            # save loss and metric for each epoch
            self.history[epoch] = {'loss': loss, 'metric': metric, 'val_loss': val_loss, 'val_metric': val_metric}

            if self.verbose:
                if val_dataset is not None:
                    print(
                        f"Epoch {epoch}/{self.epochs} - "
                        f"loss: {loss:.4f} - acc: {metric:.4f} - "
                        f"val_loss: {val_loss:.4f} - val_acc: {val_metric:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{self.epochs} - "
                        f"loss: {loss:.4f} - acc: {metric:.4f}"
                    )
            
            # early stopping
            if early_stopping and val_dataset is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_layers = [copy.deepcopy(layer) for layer in self.layers]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    if best_layers is not None:
                        self.layers = best_layers
                    break
            

        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")

