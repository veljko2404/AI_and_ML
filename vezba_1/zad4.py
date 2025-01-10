import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Na osnovu dataseta covid19, istrenirati model koji vraća smrtnost
# za zadati broj obolelih. Naći grešku, isrctati model.

data = pd.read_csv("covid19.csv")

cases = np.asarray(data["new_cases"])
deaths = np.asarray(data["deaths"])

model = LinearRegression()
model = model.fit(cases.reshape(-1, 1), deaths)

predictions = model.intercept_ + model.coef_[0] * cases.flatten()

mae = mean_absolute_error(deaths, predictions)

plt.scatter(cases, deaths)
plt.title("Mean absolute error is " + str(round(mae, 2)))
plt.xlabel("Cases")
plt.ylabel("Deaths")
plt.plot((0, 10000), (model.predict([[0]]), model.predict([[10000]])), "r")
plt.show()
