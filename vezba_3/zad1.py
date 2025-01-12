import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("data.csv")
x = np.asarray(data["x"]).reshape(-1, 1)
y = np.asarray(data["y"])

df_test = pd.read_csv("data_test.csv")
x_test = np.asarray(df_test["x"]).reshape(-1, 1)
y_test = np.asarray(df_test["y"])

degrees = [3]
als = [1, 2, 3, 5]
OLS_reg_train = []
OLS_reg_test = []
RIDGE_reg_train = []
RIDGE_reg_test = []
LASSO_reg_train = []
LASSO_reg_test = []
for degree in degrees:
    for al in als:
        x_transformed = PolynomialFeatures(degree, include_bias=False).fit_transform(x)
        x_test_transformed = PolynomialFeatures(degree, include_bias=False).fit_transform(x_test)
        model1 = LinearRegression().fit(x_transformed, y)
        model2 = Ridge(alpha=al).fit(x_transformed, y)
        model3 = Lasso(alpha=al).fit(x_transformed, y)
        plt.scatter(x, y)
        plt.scatter(x_test, y_test)
        plt.plot(np.concatenate([x, x_test]), model1.predict(np.concatenate([x_transformed, x_test_transformed])),
                 label="OLS")
        plt.plot(np.concatenate([x, x_test]), model2.predict(np.concatenate([x_transformed, x_test_transformed])),
                 label="Ridge")
        plt.plot(np.concatenate([x, x_test]), model3.predict(np.concatenate([x_transformed, x_test_transformed])),
                 label="Lasso")
        plt.legend()
        plt.show()

        print("MAE")
        print("OLS regression:")
        print("train_set:" + str(mean_absolute_error(model1.predict(x_transformed), y)))
        print("test_set:" + str(mean_absolute_error(model1.predict(x_test_transformed), y_test)))
        print("Ridge regression:")
        print("train_set:" + str(mean_absolute_error(model2.predict(x_transformed), y)))
        print("test_set:" + str(mean_absolute_error(model2.predict(x_test_transformed), y_test)))
        print("Lasso regression:")
        print("train_set" + str(mean_absolute_error(model3.predict(x_transformed), y)))
        print("test_set" + str(mean_absolute_error(model3.predict(x_test_transformed), y_test)))

        OLS_reg_train.append(mean_absolute_error(model1.predict(x_transformed), y))
        OLS_reg_test.append(mean_absolute_error(model1.predict(x_transformed), y))

        RIDGE_reg_train.append(mean_absolute_error(model2.predict(x_transformed), y))
        RIDGE_reg_test.append(mean_absolute_error(model2.predict(x_transformed), y))

        LASSO_reg_train.append(mean_absolute_error(model3.predict(x_transformed), y))
        LASSO_reg_test.append(mean_absolute_error(model3.predict(x_transformed), y))

print(OLS_reg_test)
print(OLS_reg_train)

if len(degrees) > 1:
    plt.plot(degrees, OLS_reg_train, label="train")
    plt.plot(degrees, OLS_reg_test, label="test")
    plt.title("OLS sa promenom degree")
    plt.legend()
    plt.show()

if len(als) > 1:
    plt.plot(als, RIDGE_reg_train, label="train")
    plt.plot(als, RIDGE_reg_test, label="test")
    plt.legend()
    plt.title("MAE RiDGE,train i test sa promenom alfa")
    plt.show()

    plt.plot(als, LASSO_reg_train, label="train")
    plt.plot(als, LASSO_reg_test, label="test")
    plt.title("MAE LASSO,train i test sa promenom alfa")
    plt.legend()
    plt.show()
