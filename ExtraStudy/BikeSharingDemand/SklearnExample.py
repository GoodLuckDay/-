import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("./Data/train.csv", parse_dates=["datetime"])
test = pd.read_csv("./Data/test.csv", parse_dates=["datetime"])

train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second
train["dayofweek"] = train["datetime"].dt.dayofweek

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["minute"] = test["datetime"].dt.minute
test["second"] = test["datetime"].dt.second
test["dayofweek"] = test["datetime"].dt.dayofweek

feature_names = ["season","holiday","workingday","weather",
                             "dayofweek","month","year","hour"]

for var in feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")

X_train = train[feature_names]
X_test = test[feature_names]

label_name = "count"

y_train = train[label_name]

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.ensemble import RandomForestClassifier

max_depth_list = []
model = RandomForestClassifier(n_estimators=5, n_jobs=1, random_state=0)

score = cross_val_score(model, X_train, y_train, cv=k_fold)
score = score.mean()
print("Score = {0:.5f}".format(score))
model.fit(X_train, y_train)
predictions = model.predict(X_test)