import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
import pickle

if __name__ == "__main__":

    covid = pd.read_csv("data.csv")
    # covid.info()
    # corr = covid.corr()
    # print(corr['infected'].sort_values(ascending=False))

    f = covid.drop("infected", axis = 1)
    l = covid["infected"].copy()

    model = LogisticRegression()
    model.fit(f,l)

    p = model.predict([[103,1,1,1,1,0,67]])
    prob = model.predict_proba([[103,1,1,1,1,0,67]])[0][1]

    # storing the file
    file =open ("covid.pkl", "wb")
    # dumping the file
    pickle.dump(model, file)
    file.close()