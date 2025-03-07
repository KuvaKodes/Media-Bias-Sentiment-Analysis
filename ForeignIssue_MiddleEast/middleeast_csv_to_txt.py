import pandas as pd
import re

# Dictionary mapping filenames to their respective keywords
file_keyword_map = {
    "Afghanistan_war_concordance.txt": "Afghanistan war",
    "bin_laden_concordance.txt": "Bin Laden",
    "gulf_war_concordance.txt": "Gulf war",
    "hussein_concordance.txt": "Hussein",
    "iraq_war_concordance.txt": "Iraq war",
    "ISIL_concordance.txt": "ISIL",
    "isis_concordance.txt": "ISIS",
    "islamic_state_concordance.txt": "Islamic State",
    "operation_desert_storm_concordance.txt": "Operation Desert Storm",
    "qaeda_concordance.txt": "Qaeda",
    "taliban_concordance.txt": "Taliban",
    "war_in_afghanistan_concordance.txt": "War in Afghanistan",
    "war_on_terror_concordance.txt": "War on Terror",
    "weapons_of_mass_destruction_concordance.txt": "Weapons of Mass Destruction",
    "wmd_concordance.txt": "WMD"
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
combined_df.to_csv('womens_rights_combined_data.csv', index=False)
