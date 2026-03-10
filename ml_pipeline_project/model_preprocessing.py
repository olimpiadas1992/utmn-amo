import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

train = pd.read_csv("data/train/train.csv")
test = pd.read_csv("data/test/test.csv")

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

pd.DataFrame(train_scaled).to_csv("data/train/train_scaled.csv", index=False)
pd.DataFrame(test_scaled).to_csv("data/test/test_scaled.csv", index=False)

joblib.dump(scaler, "scaler.pkl")

print("Preprocessing done")