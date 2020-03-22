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
    

    def train(self, a=0.0001, epochs=0, print_cost=False):
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

            if print_cost:
                print(new_cost)
                plt.title("Cost with learning rate "+str(a))
                plt.scatter(i, new_cost, c="r")
                plt.pause(0.0001)

        print("weights",weights)
        d_w = (np.linalg.inv((self.x.T)@self.x)@((self.x.T)@self.y) )
        print(d_w)
        print(weights)
        self.weights = weights
        return d_w

    def predict(self,*args):
        largs = list(args)
        largs.append(1)
        x = np.array(largs)
        return self.weights@x.T


#give the names of columns like ('X', 'Z')
#and then they will be transfered to a numpy matrix
def panda_to_numpy(data, *args):
    return np.column_stack([data[d] for d in args])


if __name__ == "__main__":
    data_path = "/home/andreas/Documents/regression/ph-r.xls"
    data = pd.read_excel(data_path)

    x_train = data['X']
    y_train = data['Y']
    z_train = data['Z']

    # y must be an nX1 array
    # x myst be an nXm array where m is the number of different variables for the reggression
    # regression() returns an 1Xm+1 array wich are the weights +1 is for the constant 
    x = panda_to_numpy(data, 'X', 'Z')
    y = data['Y']
    model = Regression(y,x)
    weights = model.train(epochs=200, a=0.00006, print_cost=False) 
    prediction = model.predict(8,15)
    print(prediction)


    # plot the results
    ones = np.ones(len(x_train))
    x_pred = np.column_stack(( np.linspace(x_train.min(),x_train.max(), 34), z_train, ones))
    y_pred = weights@x_pred.T


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_train, z_train, y_train, "ro")
    ax.plot( x_pred[:,0], x_pred[:,1], y_pred, "g")
    plt.show()