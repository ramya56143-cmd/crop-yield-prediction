import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset (example CSV)
df = pd.read_csv("crop_yield.csv")

#  Features & Target
X = df[['Rainfall', 'Temperature', 'Fertilizer', 'Pesticide']]
y = df['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("crop_model.pkl", "wb"))

print("Model Trained & Saved")