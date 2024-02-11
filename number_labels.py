import pandas as pd

# Read the CSV file
df = pd.read_csv("challenge_data/val.csv")

# Count the occurrences of each label
label_counts = df['labels'].value_counts()

# Print the label counts
print(label_counts)

# labels challenge_data/train.csv
# No_DR             20672
# Moderate           4213
# Mild               1961
# Severe              677
# Proliferate_DR      577
# Name: count, dtype: int64

# labels challenge_data/val.csv
# No_DR             5138
# Moderate          1079
# Mild               482
# Severe             196
# Proliferate_DR     131
# Name: count, dtype: int64