import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("data.csv")

print(data)
x = np.asarray(data["x"])
y = np.asarray(data["y"])

plt.scatter(x, y)
plt.show()

x = x.reshape(-1, 1)
for deg in [2, 3, 5, 8, 12]:
    x_transformed = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(x)
    model = LinearRegression().fit(x_transformed, y)
    plt.plot(x, model.predict(x_transformed))
    plt.scatter(x, y)
    plt.title("degree: " + str(deg) + ", MAE: " + str(mean_squared_error(y, model.predict(x_transformed))))
    plt.show()
