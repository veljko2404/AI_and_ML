import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Za skup podataka house_pricing iscrtati podatke.
# Pomocu jednacine prave definisati model koji za kvadraturu vraca cenu
# stana (rucno namestiti koeficijente). Iscrtati pravu

df = pd.read_csv("house_pricing.csv")

square_meters = np.asarray(df["sqm"])
price = np.asarray(df["price"])


def abs_error(p, p_pred):
    return np.mean(np.abs(p - p_pred))


def sqrt_error(p, p_pred):
    return np.mean((p - p_pred) ** 2)


def get_price(p):
    result = int(p) * 1000 + 0
    return result


price_pred = list(map(get_price, square_meters))

print("Absolute error is " + str(abs_error(price, price_pred)) + " eur")
print("Squared absolute error is " + str(sqrt_error(price, price_pred)) + " eur")

plt.scatter(square_meters, price)
plt.show()
