import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

train = pd.read_csv("data/train/train_scaled.csv")

X = train.iloc[:,0].values.reshape(-1,1)
y = train.iloc[:,1]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained")