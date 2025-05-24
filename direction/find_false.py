import pandas as pd

# Read the CSV file
# Replace 'your_file.csv' with the path to your actual CSV file
df = pd.read_csv('labels_train.csv')

# Filter rows where angle is greater than 360 or less than 0
invalid_angles = df[(df['angle'] > 360) ]

# Check if any rows were found
if not invalid_angles.empty:
    print("Rows with angle > 360 or < 0:")
    print(invalid_angles)
else:
    print("No rows found with angle > 360 or < 0.")