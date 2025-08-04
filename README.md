# Predictive-Tire-Maintenance
This project uses mahcine learning to predict tire failure based on 
- Tread depth
- Mileage 
- Pressure 
- Temperature
- Tire age

The goal is to demonstrate how predictive maintenance can help reduce vehicle downtime and improve safety for fleet operations or automotive technicians.

# Folder Structure'
- data - contains - tire data (Tire_Data_1000.csv)

- model contains
       - train_model.py -> Trains and evaluates the ML model
       - predict_status.py -> predicts failure from sample input (not needed)
       - status-model.py -> Trained model file

        

# How to Run
1. make sure you have Python installed
2. Install dependencies 
 '''bash
     pip install pandas scikit-learn joblib streamlit
3. run the model script
   cd model
   pyhthon train_model.py
4. run the prediction script
   pyhton predict_status.py (display a sample prediction using the model before front end)

# Running Web App
'''bash 
   streamlstreamlit run app.py (or) python -m streamlit run app.py

