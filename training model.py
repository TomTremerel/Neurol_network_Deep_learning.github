# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:04:25 2024

@author: TOM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles


def initialisation(dimensions):
    C = len(dimensions)
    parametres = {}

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c-1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres

def forward_propagation(X, parametres): 
    activations = {'A0': X}
    C = len(parametres) // 2
    for c in range(1, C + 1):
        Z = parametres['W' + str(c)].dot(activations['A'+ str(c - 1)]) + parametres['b' + str(c)]
        activations['A'+ str(c)] = (1 / (1 + np.exp(-Z)))

    return activations

def back_propagation(y, parametres, activations):

    C = len(parametres) // 2
    gradients = {}
    dZ = activations['A' + str(C)] - y
    m = y.shape[1]

    for c in reversed(range(1, C + 1)):
        gradients['dW'+ str(c)] = 1/m * dZ.dot(activations['A' + str(c-1)].T)
        gradients['db' + str(c)] = 1/m * np.sum(dZ)

        dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c-1)] * (1 - activations['A' + str(c-1)])

    return gradients

def update(parametres, gradients, learning_rate):

    C = len(parametres) // 2

    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return parametres

def predict(X, parametres): 

    C = len(parametres) // 2
    activations = forward_propagation(X, parametres)
    Af = activations['A' + str(C)]

    return Af >= 0.5

def plot_decision_boundary(X, y, parametres):
    h = 0.01
    x_min, x_max = X[0, :].min() - 0.1, X[0, :].max() + 0.1
    y_min, y_max = X[1, :].min() - 0.1, X[1, :].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parametres)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral)
    plt.show()

def neurol_network(X, y, hidden_layers, learning_rate, n_iter):

     np.random.seed(0)

     dimensions = list(hidden_layers)
     dimensions.insert(0, X.shape[0])
     dimensions.append(y.shape[0])
     parametres = initialisation(dimensions)

     train_loss = []
     train_acc = []

     for i in tqdm(range(n_iter)):
         activations = forward_propagation(X, parametres)
         gradients = back_propagation(y, parametres, activations)
         parametres = update(parametres, gradients, learning_rate)

         if i % 10 == 0:
             C = len(parametres) // 2
             train_loss.append(log_loss(y, activations['A' + str(C)]))
             y_pred = predict(X, parametres)
             current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
             train_acc.append(current_accuracy)

     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
     ax[0].plot(train_loss, label='train loss')
     ax[0].legend()

     ax[1].plot(train_acc, label='train acc')
     ax[1].legend

     plot_decision_boundary(X, y, parametres)
     plt.show()

     return parametres


X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)

X = X.T

y = y.reshape((1, y.shape[0]))

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')

neurol_network(X, y, hidden_layers=(32, 32, 32), learning_rate=0.01, n_iter=4000)


    
    