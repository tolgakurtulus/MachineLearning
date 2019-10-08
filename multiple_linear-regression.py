# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:22:23 2019

@author: tolga
"""

# multiple linear regression =   y = b0+ b1*x1 + b2*x2

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple-linear-regression-dataset.csv", sep = ";")

x = df.iloc[:,[0,2]].values  # We take the deneyim and yas because this is x
y = df.maas.values.reshape(-1,1) # We take the maas because this is x


#%%multiple linear section

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0 = ", multiple_linear_regression.intercept_ )
print("b1, b2 = ", multiple_linear_regression.coef_ )

multiple_linear_regression.predict(np.array([[3,35],[5,35]]))

