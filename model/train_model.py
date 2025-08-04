import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# File path to your CSV
file_path = '../data/Tire_Data_1000.csv'

print("📌 Script started...")
print(f"📂 Loading data from: {file_path}")

# Read the CSV with encoding fix
df = pd.read_csv(file_path, encoding='utf-8-sig')

# Clean column names
df.columns = [col.strip() for col in df.columns]
print("✅ Data loaded. Shape:", df.shape)
print("🧠 Columns found:", df.columns.tolist())

# ✅ Prepare features and labels (no more 'status' column used)
features = df[['tread_depth', 'pressure', 'mileage', 'age_months', 'temperature']]
labels = df['failure']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("\n🎯 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy Score:", round(accuracy_score(y_test, y_pred), 4))

# Save the model only
joblib.dump(model, 'status_model.pkl')
print("💾 Model saved as 'status_model.pkl'")