import pandas as pd

#Loading the combined dataset from its directory
df = pd.read_csv("data/beijing_combined.csv")



#creating required 100-sample set randomly from my combined dataset
subset = df.sample(n=100, random_state=42)

#saving the subset as subset_100.csv
subset.to_csv("data/subset_100.csv", index=False)

print("Subset created successfully.")
