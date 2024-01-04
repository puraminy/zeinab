import pandas as pd
import numpy as np

# Load data from CSV file into a DataFrame
df = pd.read_csv('raw_data.csv')
# Define a dictionary for mapping long labels to short labels
label_mapping = {
    'heat transfer Cofficent':'HTC',
    'flow rate(m3/s)': 'flow_rate',
    'concentration of nanoparticle(vol. fraction))': 'conc_nano',
    'Kfluid(W/mC)': 'Kfluid',
    'heat flux(W/m2)': 'heat_flux',
    'X/D': 'X_D',
    'heat transfer Cofficent (prediction with ANN)1': 'HTC_ANN1',
    'heat transfer Cofficent (prediction with ANN)2': 'HTC_ANN2',
}

# Replace non-standard decimal separator and convert to numeric
for long_label, short_label in label_mapping.items():
    df[short_label] = pd.to_numeric(df[long_label].astype(str).str.replace('٫', '.'), errors='coerce')

# Drop the original long-label columns
df = df.drop(columns=list(label_mapping.keys()))

# Drop rows with NaN values
df = df.dropna()
# Save the resulting DataFrame to a new CSV file
df.to_csv('data.csv', index=False)


