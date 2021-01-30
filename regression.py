import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random


class Regression():
    
    def __init__(self, y, x):
        #organise data in arrays 
        ones = np.ones(len(x))
        x = np.column_stack((x,ones))
        self.rows = x.shape[0]
        self.columns = x.shape[1]
        self.y = y
        self.x = x
    

    def train(self, a=0.0001, epochs=0, print_loss=False):
        print("Start regression with input of",self.columns-1,"variables")

        # initialize weights random
        weights = np.random.rand(self.columns)

        # initialize derivatives
        derivatives = np.random.rand(self.columns)

        # cost function
        def cost(y,x,w):
            return sum((y - (w@x.T))**2)

        # train process
        for i in range(epochs):

            #update weights
            weights[:] = [weight - a*derivative for weight,derivative in zip(weights,derivatives)]

            #update derivatives
            derivatives[:] = [-sum((self.y- weights@self.x.T)*2*row) for row in self.x.T]
            # print(derivatives)
            new_cost = cost(self.y,self.x,weights)

            if print_loss:
                print(new_cost/self.rows)
                plt.title("Loss with learning rate "+str(a))
                plt.ylabel("Loss")
                plt.xlabel("Epoch")
                plt.scatter(i, new_cost/self.rows, c="r")
                plt.pause(0.0001)

        print("weights",weights)
        self.weights = weights
        return self.weights

    def predict(self,*args):
        largs = list(args)
        largs.append(1)
        x = np.array(largs)
        return self.weights@x.T


#give the names of columns like ('X', 'Z')
#and then they will be transfered to a numpy matrix
def panda_to_numpy(data, *args):
    return np.column_stack([data[d] for d in args])

