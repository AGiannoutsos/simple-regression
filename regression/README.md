# Regression

datasets are used from http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html 


### How to run

run any `python3 health.py` , `bicarbonate.py` or `hollywood.py` for the already implemented datasets

In `health.py` dataset contents are X1 = death rate per 1000 residents
X2 = doctor availability per 100,000 residents
X3 = hospital availability per 100,000 residents
X4 = annual per capita income in thousands of dollars
X5 = population density people per square mile

In `bicarbonate.py` dataset X = pH of well water
Y = Bicarbonate (parts per million) of well water

In `hollywood.py` dataset X1 = first year box office receipts/millions
X2 = total production costs/millions
X3 = total promotional costs/millions

### How to run any model

The regression.py file implements the Regression class where it can calculate a problem.
The data is entered in the class initialization `model = Regression (y, x)`

To enter the data `y` must be an NumPy array of n lines and 1 column and this will be the target of our prediction.

`x` should be a table of NumPy n lines equal to `y` and m columns where the different attributes will be for a Multiple Variable Regression


After the data entry, the model needs training
For training we use the `train (a, epochs, print_loss)` function where `a` is the learning rate `epochs` are the iterations of the learning process
`print_loss` is a True or False value if we want to print it in real time the loss of the regression


And finaly to predict the results we use the predict function `predict(x1, x2 ...)`
which input are the X different attributes from which we want to arrive at a prediction Y


### Visualization

Hollywood movies revenue prediction based on production and promotion costs
![](https://github.com/AGiannoutsos/ai-projects/blob/master/regression/hollywood.png)

and loss in epochs

![](https://github.com/AGiannoutsos/ai-projects/blob/master/regression/hollywood_loss.png)


Prediction in ppm of bicarbonates based on ph of water

![](https://github.com/AGiannoutsos/ai-projects/blob/master/regression/biocarbonate.png)


