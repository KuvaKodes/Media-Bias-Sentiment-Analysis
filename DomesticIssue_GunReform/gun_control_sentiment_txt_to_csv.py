import pandas as pd
import re

# Dictionary mapping filenames to their respective keywords
file_keyword_map = {
    "gun_control_concordance.txt": "gun control",
    "assault_ban_concordance.txt": "assault weapons ban",
    "second_amendment_concordance.txt": "second amendment",
    "NRA_concordance.txt": "NRA",
    "gun_legislation_concordance.txt": "gun legislation",
    "gun_reform_concordance.txt": "gun reform",
    "brady_bill_concordance.txt": "brady bill",
    "firearm_concordance.txt": "firearm",
    "gun_rights_concordance.txt": "gun rights",
    "gun_show_concordance.txt": "gun show",
    "gun_violence_concordance.txt": "gun violence",
    "mass_shooting_concordance.txt": "mass shooting"
}

# Initialize an empty list to collect DataFrames
dfs = []

for filename, keyword in file_keyword_map.items():
    # Attempt to read the file with 'python' engine and skipping bad lines
    # Adjust the parameters if necessary depending on the actual file format.
    df = pd.read_csv(filename, sep='\t', encoding='utf-8', engine='python', on_bad_lines='skip')
    
    # Add the filename and keyword as new columns
    df['filename'] = filename
    df['keyword'] = keyword
    
    # Append this df to the list
    dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Optional: clean up whitespace in text columns if needed
text_columns = ["Context before", "Query item", "Context after"]
for col in text_columns:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# Write out to a single CSV file
combined_df.to_csv('combined_gun_control_all_data.csv', index=False)
