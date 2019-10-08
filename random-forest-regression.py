# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:33:43 2019

@author: tolga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("random-forest-regression-dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%  Random Forest REg.  Sec. Predict Sec

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state=42)
rf.fit(x,y)

print("12 Level Price = ", rf.predict([[7]]))

#%%  Visu.. Sec.

x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x,y, color="red")
plt.plot(x_,y_head,color="black")
plt.ylabel("Tiribün")
plt.xlabel("Money")
plt.show()