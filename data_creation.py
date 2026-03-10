import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_samples = 200

days = np.arange(n_samples)
temperature = 20 + 10*np.sin(days/10) + np.random.normal(0, 1, n_samples)

# добавим аномалии
temperature[::25] += 10

data = pd.DataFrame({
    "day": days,
    "temperature": temperature
})

train = data[:150]
test = data[150:]

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

train.to_csv("data/train/train.csv", index=False)
test.to_csv("data/test/test.csv", index=False)

print("Data created")