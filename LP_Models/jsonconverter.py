import pandas as pd

# Replace 'your_file_path.csv' with the actual path to your CSV file
xlsx_file_path = 'C:\\Users\\parkway\\OneDrive - Bluetag Group\\Desktop\\testdata\\testdata 2.xlsx'


df = pd.read_excel(xlsx_file_path)

# Convert DataFrame to JSON
json_data = df.to_json(orient='records')

# Print the JSON data
print(json_data)
