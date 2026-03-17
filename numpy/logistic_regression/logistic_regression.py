import numpy as np
from src.dataset import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re

def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

class LogisticRegression:

    def __init__(self, dataset): 
        
        print(dataset.X.shape)
        self.X = np.hstack((np.ones([dataset.nrows(),1]), dataset.X))
        
        self.y = dataset.Y
        self.n_classes = len(np.unique(self.y))
        self.y_onehot = self.one_hot(self.y)
        self.theta =  np.zeros((self.X.shape[1], self.n_classes))
        self.data = dataset

    def one_hot(self, y):
        onehot = np.zeros((len(y), self.n_classes))

        for i, val in enumerate(y):
            onehot[i, int(val)] = 1

        return onehot

    def probability(self, instance):

        x = np.empty([self.X.shape[1]])
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        
        z = np.dot(x, self.theta)

        return softmax(z)
    
    def predict(self, X, y, num_rows):

        correctGuesses = 0
        
        for n in range(num_rows):
            probs = self.probability(X[n])
            result = np.argmax(probs)
            if result == y[n]:
                correctGuesses += 1

        return correctGuesses / num_rows * 100
    
    def costFunction(self, theta = None):
        if theta is None: theta = self.theta
        m = self.X.shape[0]
        z = self.X.dot(self.theta)
        probs = np.apply_along_axis(softmax, 1, z)
        cost = -np.sum(self.y_onehot * np.log(probs)) / m
        return cost
    
    def gradientDescent(self, alpha = 0.01, iters = 10000):
        m = self.X.shape[0]
        for its in range(iters):
            z = self.X.dot(self.theta)
            probs = np.apply_along_axis(softmax, 1, z)
            gradient = self.X.T.dot(probs - self.y_onehot) / m
            self.theta -= alpha * gradient
            if its % 1000 == 0: print(self.costFunction())
                

    def buildModel(self):
        self.optim_model()

    def optim_model(self):
        from scipy import optimize
            
        n = self.X.shape[1]
        options = {'full_output': True, 'maxiter': 500}
        initial_theta = np.zeros(n)
        self.theta, _, _, _, _ = optimize.fmin(lambda theta: self.costFunction(theta, initial_theta, **options))

    def printCoefs(self):
        print(self.theta)

    def plotModel(self):
        from numpy import r_
        pos = (self.y == 1).nonzero()[:1]
        neg = (self.y == 0).nonzero()[:1]
        plt.plot(self.X[pos, 1].T, self.X[pos, 2].T, 'k+', markeredgewidth = 2, markersize = 7)
        plt.plot(self.X[neg,1].T, self.X[neg, 2]. T, 'ko', markeracecolor = 'r', markersize = 7)
        if self.X.shape[1] <= 3:
            plot_x = r_[self.X[:,2]. min(), self.X[:,2].max()]
            plot_y = (-1./self.theta[2]) * (self.theta[1]*plot_x + self.theta[0])
            plt.plot(plot_x, plot_y)
            plt.legend(['class 1', 'class 0', 'Decision Boundary'])
        plt.show()


