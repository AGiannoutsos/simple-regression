import regression as r
import pandas as pd

data_path = "/home/andreas/Documents/regression/mlr07.xls"
data = pd.read_excel(data_path)


# y must be an nX1 array
# x myst be an nXm array where m is the number of different variables for the reggression
# regression() returns an 1Xm+1 array wich are the weights +1 is for the constant 
x = r.panda_to_numpy(data, 'X2', 'X3','X4','X5',)
y = data['X1']
model = r.Regression(y,x)
weights = model.train(epochs=2000, a=0.00000001, print_cost=False) 
prediction = model.predict(50,200,5,150)
print(prediction)