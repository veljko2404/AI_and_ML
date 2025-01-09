import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge

# Dodati novi podatak: kvadratura stana 60 I cena stana 20 000.
# Istrenirati model na tim podacima. Iscrtati regresionu liniju
# za tako dobijen model, i prethodno istreniran model.

# Iskoristiti Ridge regressiju sa alpha faktorom = 1000. Iscrtati sva tri modela

df = pd.read_csv("house_pricing.csv")

square_meters = np.asarray(df["sqm"])
price = np.asarray(df["price"])
square_meters_2 = np.append(square_meters, 60)
price_2 = np.append(price, 20000)

slope, intercept, r, p, std_error = stats.linregress(square_meters, price)

slope_2, intercept_2, r_2, p_2, std_error_2 = stats.linregress(square_meters_2, price_2)


def get_price(x):
    return x * slope + intercept


def get_price_2(x):
    return x * slope_2 + intercept_2


data = list(map(get_price, square_meters))
data_2 = list(map(get_price_2, square_meters_2))
data_ridge = Ridge(alpha=1000)
data_ridge = data_ridge.fit(square_meters_2.reshape(-1, 1), price_2)

plt.scatter(square_meters_2,price_2)
plt.plot(square_meters, data, label="Data")
plt.plot(square_meters_2, data_2, label="Data 2")
plt.plot((10, 70), (data_ridge.predict([[10]]), data_ridge.predict([[70]])), label="Ridge")
plt.legend()
plt.show()
