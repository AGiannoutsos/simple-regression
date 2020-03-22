import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from regression import Regression, panda_to_numpy

data_path = "./biocarbonate.xls"
data = pd.read_excel(data_path)

x_train = data['X']
y_train = data['Y']

# y must be an nX1 array
# x myst be an nXm array where m is the number of different variables for the reggression
# regression() returns an 1Xm+1 array wich are the weights +1 is for the constant 
x = panda_to_numpy(data, 'X')
y = data['Y']
model = Regression(y,x)
weights = model.train(epochs=90000, a=0.00005, print_loss=False) 
prediction = model.predict(8)
print(prediction)


# plot the results
ones = np.ones(len(x_train))
x_pred = np.column_stack(( np.linspace(x_train.min(),x_train.max(), len(x_train)), ones))
y_pred = weights@x_pred.T


plt.title("Prediction of bicarbonate")
plt.xlabel("ph")
plt.ylabel("biocarbonates ppm")
plt.plot(x_train, y_train, "ro", x_pred, y_pred, "g--")
plt.axis([x_train.min(),x_train.max(), y_train.min(),y_train.max()])
plt.show()