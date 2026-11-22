import pandas as pd

df = pd.read_csv("data/Building_Permits.csv", low_memory=False)
df_clean = df.copy()

print("Shape:", df_clean.shape)
print("\n--- Q1: Head ---")
print(df_clean.head())

print("\n--- Q2: Info ---")
df_clean.info()

print("\n--- Q3: Permit Number duplicates ---")
print(df_clean["Permit Number"].duplicated().value_counts())

df_clean.rename(
    columns={
        "Permit Creation Date": "Date Created",
        "Street Number": "House Number",
    },
    inplace=True,
)
print("\n--- Q4: Renamed columns check ---")
print([c for c in ["Date Created", "House Number"] if c in df_clean.columns])

missing_pct = (df_clean.isna().sum().sum() / df_clean.size) * 100
print(f"\n--- Q5: Missing percentage: {missing_pct:.2f}% ---")

print("\n--- Q6: Missing in Street Number Suffix & Zipcode ---")
print("Street Number Suffix:", df_clean["Street Number Suffix"].isna().sum())
print("Zipcode:", df_clean["Zipcode"].isna().sum())

print("\n--- Q7: Drop rows with any NA ---")
print("Shape after dropna(how=\"any\"):", df_clean.dropna(how="any").shape)

print("\n--- Q8: Drop columns with any vs all NA ---")
print("Shape dropna(how=\"any\", axis=1):", df_clean.dropna(how="any", axis=1).shape)
print("Shape dropna(how=\"all\", axis=1):", df_clean.dropna(how="all", axis=1).shape)

df_filled = df_clean.bfill().fillna(0)
print("\n--- Q9: Remaining NAs after bfill + fillna(0):", df_filled.isna().sum().sum())

df_filtered = df_clean[
    ~df_clean["Permit Type Definition"].fillna("").str.contains("sign - erect", case=False, regex=False)
].copy()
print("\n--- Q10: Shape after removing \"sign - erect\":", df_filtered.shape)
