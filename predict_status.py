import pandas as pd
import joblib

# Load model and encoder
model = joblib.load('status_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Print available status classes for reference
print("🔠 Valid status values:", list(label_encoder.classes_))

# Sample Input

sample = {
    'tread_depth': 5.8,       
    'pressure': 105.0,         # PSI
    'mileage': 45000,          # miles
    'age_months': 30,          # months
    'temperature': 150.0,      # °F
    'status': 'Inspect'          
}

# Validate status
if sample['status'] not in label_encoder.classes_:
    raise ValueError(f"❌ Invalid status '{sample['status']}'. Valid options: {list(label_encoder.classes_)}")

#  status
sample['status_encoded'] = label_encoder.transform([sample['status']])[0]
del sample['status']

# data order
df = pd.DataFrame([[
    sample['tread_depth'],
    sample['pressure'],
    sample['mileage'],
    sample['age_months'],
    sample['temperature'],
    sample['status_encoded']
]], columns=['tread_depth', 'pressure', 'mileage', 'age_months', 'temperature', 'status_encoded'])

# Predict
prediction = model.predict(df)[0]
proba = model.predict_proba(df)[0][1]  # Probability 

# Output
print(f"🔮 Predicted Failure: {prediction} (1 = Fail, 0 = No Fail)")
print(f"📊 Probability of Failure: {proba:.2f}")
