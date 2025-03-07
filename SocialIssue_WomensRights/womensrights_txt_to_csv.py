import pandas as pd
import re

# Dictionary mapping filenames to their respective keywords
file_keyword_map = {
    "abortion_concordance.txt": "abortion",
    "equal_pay_concordance.txt": "equal pay",
    "feminism_concordance.txt": "feminism",
    "feminist_concordance.txt": "feminist",
    "gender_discrimination_concordance.txt": "gender discrimination",
    "gender_equality_concordance.txt": "gender equality",
    "glass_ceiling_concordance.txt": "glass ceiling",
    "htmetoo_concordance.txt": "#metoo",
    "metoo_concordance.txt": "metoo",
    "pay_equity_concordance.txt": "pay equity",
    "reproductive_rights_concordance.txt": "reproductive rights",
    "sex_discrimination_concordance.txt":"sex discrimination",
    "title9_concordance.txt": "title ix",
    "womens_rights_concordance.txt": "women 's rights"

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
