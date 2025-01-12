import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("monthly_consumption.txt", header=None)

x = data.iloc[:, 0]
y = data.iloc[:, 1]


def predict(month):
    return np.sin(month) * 1000 + 1500


plt.scatter(x, y)
plt.plot(x, predict(x))
plt.show()

x_transformed = np.sin(x)
plt.scatter(x_transformed, y)
plt.show()

y_predict = x_transformed * 1000 + 1500
plt.scatter(x_transformed, y)
plt.plot(x_transformed, y_predict)
plt.show()
