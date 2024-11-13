import pandas as pd
import os

# Specify the base directory containing the CSV files
base_directory = "./"

# Desired column order
desired_columns = ["average", "dogs", "cats", "lions", "chairs", "goats", "cows", "cherries", "roses", "boats", "ref"]

for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            
            # Remove rows where 'ref' column value is 'laptops'
            try:
                df = df[df['ref'] != 'laptops']
            except:
                pass
            
            # If 'laptops' column exists, remove it and then reorder the columns
            if 'laptops' in df.columns:
                df = df.drop(columns=['laptops'])
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            # Reorder the columns if they are as specified in the desired_columns list
            if set(df.columns) == set(desired_columns):
                df = df[desired_columns]
                print(f"Saving {file_path}")
                # Save the modified DataFrame back to the original file
                df.to_csv(file_path, index=False)

# Function to reorder columns and save the file
def process_file(file_path,desired_order):
    df = pd.read_csv(file_path)
    
    # Check if columns match the desired order (ignoring the order)
    if set(df.columns) == set(desired_order):
        # Reorder the columns
        df = df[desired_order]
        # Save the DataFrame back to the original file
        df.to_csv(file_path, index=False)
        print(f"Processed and saved: {file_path}")
    else:
        print(f"Column names do not match for: {file_path}")
