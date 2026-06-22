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


print("Humidity > 100 → ", (data['rel_humidity_pct'] > 100).sum(), "rows")
print("visibility_range_km < 0 → ", (data['visibility_range_km'] < 0).sum(), "rows")
print("uv_radiation_idx < 0 → ", (data['uv_radiation_idx'] < 0).sum(), "rows")

print("wind_velocity_kmh > 150 → ", (data['wind_velocity_kmh'] > 150).sum(), "rows")
print("ambient_temp_c > 60 → ", (data['ambient_temp_c'] < 60).sum(), "rows")
