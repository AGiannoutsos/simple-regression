import regression as r
import pandas as pd

data_path = "./health.xls"
data = pd.read_excel(data_path)


x = r.panda_to_numpy(data, 'X2', 'X3','X4','X5',)
y = data['X1']
model = r.Regression(y,x)
weights = model.train(epochs=2000, a=0.00000001, print_loss=False) 
prediction = model.predict(50,200,5,150)
print(prediction)