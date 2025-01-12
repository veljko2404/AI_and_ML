import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


data = pd.read_csv("data.csv")

x = np.asarray(data["x"])
y = np.asarray(data["y"])

x = x.reshape(-1, 1)

deg = 2
x_transformed = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(x)
model = LinearRegression().fit(x_transformed, y)

x1_range = np.linspace(x_transformed[:, 0].min() - 3, x_transformed[:, 0].max() + 3, 50)
x2_range = np.linspace(x_transformed[:, 1].min() - 3, x_transformed[:, 1].max() + 3, 50)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

x1_coef, x2_coef = model.coef_

intercept = model.intercept_
y_grid = x1_coef * x1_grid + x2_coef * x2_grid

fig=plt.figure(figsize=(20,30))
ax=fig.add_subplot(211,projection="3d")
ax.scatter(x_transformed[:,0],x_transformed[:,1],y,alpha=1,color="red",s=2000)
ax.view_init(elev=30,azim=85)

ax2=fig.add_subplot(212)
ax.scatter(x,y,color="red")
ax2.plot(x,model.predict(x_transformed))

plt.show()
