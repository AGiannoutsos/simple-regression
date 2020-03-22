import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from regression import Regression, panda_to_numpy

data_path = "./hollywood.xls"
data = pd.read_excel(data_path)

x_train = data['X2']
y_train = data['X1']
z_train = data['X3']

# y must be an nX1 array
# x myst be an nXm array where m is the number of different variables for the reggression
# regression() returns an 1Xm+1 array wich are the weights +1 is for the constant 
x = panda_to_numpy(data, 'X2', 'X3')
y = data['X1']
model = Regression(y,x)
weights = model.train(epochs=200, a=0.0001, print_loss=True) 
prediction = model.predict(8,15)
print(prediction)


# plot the results
ones = np.ones(len(x_train))
x_normalized = np.linspace(x_train.min(),x_train.max(), len(x_train))
z_normalized = np.linspace(z_train.min(),z_train.max(), len(z_train))
x_pred = np.column_stack(( x_normalized, z_normalized, ones))
y_pred = weights@x_pred.T


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Prediction of hollywood movie revenue")
ax.set_zlabel("Revenue")
ax.set_xlabel("Cost of production")
ax.set_ylabel("Cost of marketing")
ax.plot(x_train, z_train, y_train, "ro")
ax.plot( x_pred[:,0], x_pred[:,1], y_pred, "g")
plt.show()