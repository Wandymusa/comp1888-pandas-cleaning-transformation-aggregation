import re
import pandas as pd

movies = pd.read_csv("data/movies.csv")
m_df = movies.copy()

print("Initial shape:", m_df.shape)
print("\n--- Q1: Head ---")
print(m_df.head())

print("\n--- Q2: Info ---")
m_df.info()

dup_rows = m_df.duplicated().sum()
print(f"\n--- Q3: Exact duplicate rows: {dup_rows} ---")
m_df = m_df.drop_duplicates()

dup_movies = m_df.duplicated(subset=["MOVIES"]).sum()
print(f"--- Q4: Duplicate MOVIES rows: {dup_movies} ---")
m_df = m_df.drop_duplicates(subset=["MOVIES"])
print("Shape after dedup:", m_df.shape)

print("\n--- Q5: Missing values per column ---")
print(m_df.isna().sum())

m_df = m_df.rename(columns={"RunTime": "RUNTIME", "Gross": "GROSS"})
print("\n--- Q6: Columns after rename ---")
print(list(m_df.columns))

# Q7: Clean redundant newlines/whitespace using map, transform, and apply
m_df["GENRE"] = m_df["GENRE"].map(lambda s: s.replace("\n", "").strip() if isinstance(s, str) else s)
m_df["ONE-LINE"] = m_df["ONE-LINE"].transform(
    lambda s: s.replace("\n", "").strip() if isinstance(s, str) else s
)
m_df["STARS"] = m_df["STARS"].apply(
    lambda s: s.replace("\n", "").strip() if isinstance(s, str) else s
)

# Q8: Extract DIRECTORS and STARS
def extract_directors(val):
    if not isinstance(val, str):
        return ""
    first_part = val.split("|")[0].strip()
    if re.match(r"^(Director|Directors):", first_part, flags=re.IGNORECASE):
        return first_part.split(":", 1)[1].strip()
    return ""

def extract_stars(val):
    if not isinstance(val, str):
        return ""
    parts = [p.strip() for p in val.split("|")]
    for part in parts:
        if re.match(r"^(Star|Stars):", part, flags=re.IGNORECASE):
            return part.split(":", 1)[1].strip()
    if re.match(r"^(Director|Directors):", val.strip(), flags=re.IGNORECASE):
        return ""
    return val.strip()

m_df["DIRECTORS"] = m_df["STARS"].apply(extract_directors)
m_df["STARS"] = m_df["STARS"].apply(extract_stars)
print("\n--- Q8: Sample DIRECTORS & STARS ---")
print(m_df[["MOVIES", "DIRECTORS", "STARS"]].head())

# Q9: GROSS -> fillna(0) and float
m_df["GROSS"] = (
    m_df["GROSS"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .replace({"nan": None, "None": None, "": None})
)
m_df["GROSS"] = pd.to_numeric(m_df["GROSS"], errors="coerce").fillna(0.0).astype(float)

# Q10: RUNTIME -> fillna(mean)
m_df["RUNTIME"] = pd.to_numeric(m_df["RUNTIME"], errors="coerce")
m_df["RUNTIME"] = m_df["RUNTIME"].fillna(m_df["RUNTIME"].mean())

# Q11 & Q13: RATING -> fillna(median)
m_df["RATING"] = pd.to_numeric(m_df["RATING"], errors="coerce")
m_df["RATING"] = m_df["RATING"].fillna(m_df["RATING"].median())

# Q12: VOTES -> int, fillna with average of min and max
m_df["VOTES"] = (
    m_df["VOTES"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .replace({"nan": None, "None": None, "": None})
)
m_df["VOTES"] = pd.to_numeric(m_df["VOTES"], errors="coerce")
votes_midpoint = (m_df["VOTES"].min() + m_df["VOTES"].max()) / 2
m_df["VOTES"] = m_df["VOTES"].fillna(votes_midpoint).round().astype(int)

# Q14: Standardise YEAR to y1-y2 / y1- / -y2 format and fill nulls with NA
def standardise_year(val):
    if pd.isna(val):
        return "NA"
    text = str(val).strip()
    if not text or text.lower() == "nan":
        return "NA"
    years = re.findall(r"\b\d{4}\b", text)
    if len(years) >= 2:
        return f"{years[0]}-{years[1]}"
    if len(years) == 1:
        y = years[0]
        if re.search(rf"\b{y}\s*[-–]", text):
            return f"{y}-"
        if re.search(rf"[-–]\s*\b{y}\b", text):
            return f"-{y}"
        return y
    return "NA"

print("\n--- Q14: Unique raw YEAR (first 25) ---")
print(m_df["YEAR"].unique()[:25])
m_df["YEAR"] = m_df["YEAR"].apply(standardise_year)
print("\n--- Q14: Unique standardised YEAR (first 25) ---")
print(m_df["YEAR"].unique()[:25])
print("\nFinal dtypes:\n", m_df.dtypes)
