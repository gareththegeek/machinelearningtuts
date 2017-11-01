import pandas as pd
import numpy as np

data = pd.read_csv(".\\data\\finance\\train.csv", names=["A","B"], sep=",",header=0)

googData = np.array(data["A"],np.float)
nasdaqData = np.array(data["B"],np.float)

from sklearn.linear_model import SGDRegressor, LinearRegression

reg = SGDRegressor(eta0 = 0.1, n_iter=100000, fit_intercept=False)

reg.fit(nasdaqData.reshape(-1,1),(googData))

print(reg.coef_)

