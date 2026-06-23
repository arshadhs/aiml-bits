# Random Forest - Decision Tree

import numpy as np
import pandas as pd
import matplotlib as plot

data = pd.read_excel("PS 3.xlsx")

print (data)
print("\n" + "=" * 50)
print(data.dtypes)
print("\n" + "=" * 50)
print(data.head())
print("\n" + "=" * 50)
print(data.nunique())

def print_outliers(txt):
    print("\n" + "=" * 50)
    print (f"Data {txt}")
    print("ambient_temp_c < -60 → ", (data['ambient_temp_c'] < -60).sum(), "rows")
    print("ambient_temp_c > 60 → ", (data['ambient_temp_c'] > 60).sum(), "rows")

    print("rel_humidity_pct < 0 → ", (data['rel_humidity_pct'] < 0).sum(), "rows")
    print("rel_humidity_pct > 100 → ", (data['rel_humidity_pct'] > 100).sum(), "rows")

    print("wind_velocity_kmh > 150 → ", (data['wind_velocity_kmh'] > 150).sum(), "rows")

    print("precip_intensity_pct < 0 → ", (data['precip_intensity_pct'] < 0).sum(), "rows")
    print("precip_intensity_pct > 100 → ", (data['precip_intensity_pct'] > 100).sum(), "rows")

    print("atm_pressure_hpa  < 850 → ", (data['atm_pressure_hpa'] < 850).sum(), "rows")
    print("atm_pressure_hpa  > 1090 → ", (data['atm_pressure_hpa'] > 1090).sum(), "rows")

    print("uv_radiation_idx < 0 → ", (data['uv_radiation_idx'] < 0).sum(), "rows")
    print("uv_radiation_idx > 15 → ", (data['uv_radiation_idx'] > 15).sum(), "rows")

    print("visibility_range_km < 0 → ", (data['visibility_range_km'] < 0).sum(), "rows")
    print("visibility_range_km > 100 → ", (data['visibility_range_km'] > 100).sum(), "rows")

print_outliers("Analysis")

'''
ambient_temp_c < -60 →  0 rows
ambient_temp_c > 60 →  12975 rows
rel_humidity_pct < 0 →  0 rows
rel_humidity_pct > 100 →  416 rows
wind_velocity_kmh > 150 →  0 rows
precip_intensity_pct < 0 →  0 rows
precip_intensity_pct > 100 →  392 rows
atm_pressure_hpa  < 850 →  141 rows
atm_pressure_hpa  > 1090 →  343 rows
uv_radiation_idx < 0 →  0 rows
uv_radiation_idx > 15 →  0 rows
visibility_range_km < 0 →  0 rows
visibility_range_km > 100 →  0 rows
'''

# Convert only the rows that were erroneously left in Fahrenheit
data['ambient_temp_c'] = data['ambient_temp_c'].astype(float)

# data.loc[data['ambient_temp_c'] > 60, 'ambient_temp_c'] = (data.loc[data['ambient_temp_c'] > 60, 'ambient_temp_c'] - 32.0) * 5.0 / 9.0

# Define a clean conversion logic
def fix_temperature(temp):
    if temp > 60:
        return (temp - 32) * 5 / 9
    return temp

# Apply the function safely to the whole column
data['ambient_temp_c'] = data['ambient_temp_c'].apply(fix_temperature)

# Cap percentages at 100%
data.loc[data['rel_humidity_pct'] > 100, 'rel_humidity_pct'] = 100
data.loc[data['precip_intensity_pct'] > 100, 'precip_intensity_pct'] = 100

# Set impossible pressure readings to NaN so they don't break averages
data.loc[(data['atm_pressure_hpa'] < 850) | (data['atm_pressure_hpa'] > 1090), 'atm_pressure_hpa'] = np.nan

print_outliers("Cleaning")

print("\n" + "=" * 50)
print(data.head())

# Separate Input and Target
x = data.drop(columns=['env_condition_label (Target)'])
y = data['env_condition_label (Target)']

# One-hot encoding : convert categorical variables into binary indicator variables
categorical_cols = ['cloud_state', 'annual_phase', 'terrain_category']
x = pd.get_dummies(x, columns = categorical_cols, drop_first=True)

# Label Encoder : Convert target text labels into numbers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split Test & Train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=0.2,      # Allocates 20% of data to the test set
        random_state=42,    # Controls the randomness
        stratify=y          # Preserves the class proportions of y
    )
 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
        n_estimators = 100, # 100 district decision tree)
        random_state = 42   # Controls the randomness
    )

# Train    
model.fit(x_train, y_train)    
   
# Predict  
y_pred = model.predict(x_test)

# Model Accuracy
from sklearn.metrics import accuracy_score, classification_report
print(f"Overall Model Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Random Forest: which features drive prediction?
importances = pd.Series(model.feature_importances_, index = x.columns)
print(importances.sort_values(ascending=False))

'''
# Custom Prediction

custom_data = {
    'ambient_temp_c': 15.0,
    'rel_humidity_pct': 85.0,
    'wind_velocity_kmh': 22.0,
    'precip_intensity_pct': 75.0,
    'cloud_state': 'overcast',
    'atm_pressure_hpa': 1002.4,
    'uv_radiation_idx': 1,
    'annual_phase': 'Winter',
    'visibility_range_km': 3.0,
    'terrain_category': 'mountain'
}
custom_row = pd.DataFrame([custom_data])

custom_row_encoded = pd.get_dummies(custom_row, columns=categorical_cols, drop_first=True)
custom_row_ready = custom_row_encoded.reindex(columns=x_train.columns, fill_value=0)
# custom_row = custom_row[x_train.columns]

num_prediction = model.predict(custom_row)[0]
text_prediction = label_encoder.inverse_transform([num_prediction])[0]
print(f"Predicted Weather is: {text_prediction}")
'''
