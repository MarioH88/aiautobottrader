model = RandomForestClassifier(random_state=42, min_samples_leaf=1, max_features='auto')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your training data
# Replace 'train_data.csv' with your actual CSV file path
# The CSV should have columns: sma_20, ema_20, rsi_14, macd, macd_signal, bb_high, bb_low, target

df = pd.read_csv('train_data.csv')

# Features and target
X = df[['sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'bb_high', 'bb_low']]
y = df['target']

# Train the model

model.fit(X, y)

# Save the model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
