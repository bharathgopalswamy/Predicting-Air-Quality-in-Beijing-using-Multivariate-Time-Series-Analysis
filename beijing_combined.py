import os
import glob
import pandas as pd

# Defining the path to the folder that contains all station CSV files
folder = os.path.join(
    "data",
    "beijing_air",
    "PRSA2017_Data_20130301-20170228",
    "PRSA_Data_20130301-20170228"
)

# Finding all CSV files inside the parent folder
csv_files = glob.glob(os.path.join(folder, "*.csv"))

dataframes = []

for file in csv_files:
    df = pd.read_csv(file)

  
    if "station" not in df.columns:
        station_name = os.path.basename(file).split("_")[2]
        df["station"] = station_name

    dataframes.append(df)
# Combining all station files into a single dataframe
combined = pd.concat(dataframes, ignore_index=True)

# Saving the combined dataset as a single csv
output_path = os.path.join("data", "beijing_combined.csv")
combined.to_csv(output_path, index=False)

#For confirmation and dataset size
print("Combined dataset")
print("Rows:", combined.shape[0])
print("Columns:", combined.shape[1])
