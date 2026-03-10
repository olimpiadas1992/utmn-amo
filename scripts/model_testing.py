import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

test = pd.read_csv("data/test/test_scaled.csv")

X_test = test.iloc[:,0].values.reshape(-1,1)
y_test = test.iloc[:,1]

model = joblib.load("model.pkl")

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)

print("MSE:", mse)