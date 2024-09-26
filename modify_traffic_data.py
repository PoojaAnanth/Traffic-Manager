import pandas as pd

# Load the original dataset
traffic_data = pd.read_csv('traffic_data.csv')

# Define peak hours (e.g., between 8:00 AM to 10:00 AM)
peak_hours = (traffic_data['Time'] >= '08:00:00') & (traffic_data['Time'] <= '10:00:00')

# Increase vehicle count by 20% during peak hours
traffic_data.loc[peak_hours, 'Vehicle_Count'] *= 1.2

# Save the modified dataset to a new file
traffic_data.to_csv('traffic_data_modified.csv', index=False)

print("Dataset has been updated and saved as traffic_data_modified.csv")

