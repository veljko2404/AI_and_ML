import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Korsiteći gotove biblioteke u Python-u pomoću linearne regresije
# odrediti cenu stana na osnovu kvadrature

df = pd.read_csv("house_pricing.csv")

square_meters = np.asarray(df["sqm"])
price = np.asarray(df["price"])

slope, intercept, r, p, std_error = stats.linregress(square_meters, price)


def calculate_price(p):
    return p * slope + intercept


calc_data = list(map(calculate_price, square_meters))

plt.scatter(square_meters, price)
plt.plot(square_meters, calc_data)
plt.show()
