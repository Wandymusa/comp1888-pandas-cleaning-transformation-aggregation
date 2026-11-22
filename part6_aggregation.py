import pandas as pd

print("=== PRODUCTS: apply() vs transform() ===")
product = pd.read_csv("data/products.csv")
print("Products shape:", product.shape)
print(product.head())

prod_copy = product.copy()

mean_purchase = (
    product.groupby("User_ID")["Purchase"]
    .apply("mean")
    .rename("Purchase_mean")
    .reset_index()
)
product = product.merge(mean_purchase, on="User_ID", how="left")

prod_copy["Purchase_mean"] = prod_copy.groupby("User_ID")["Purchase"].transform("mean")

print("Comparison sample (prod_copy):")
print(prod_copy[["User_ID", "Purchase", "Purchase_mean"]].head())
print("Max diff between merge(apply) and transform:", (product["Purchase_mean"] - prod_copy["Purchase_mean"]).abs().max())

print("\n=== F500 ANALYSIS ===")
companies = pd.read_csv("data/f500.csv")
print("F500 shape:", companies.shape)
print("\n--- Q1: Head ---")
print(companies.head())

print("\n--- Q2: Info ---")
companies.info()

print("\n--- Q3: Max & Min profit ---")
max_profit = companies.loc[companies["profits"].idxmax()]
min_profit = companies.loc[companies["profits"].idxmin()]
print("Max profit:", max_profit["company"], "|", max_profit["profits"])
print("Min profit:", min_profit["company"], "|", min_profit["profits"])

print("\n--- Q4: Industry count per sector (ascending) ---")
sector_counts = companies.groupby("sector")["industry"].count().sort_values(ascending=True)
print(sector_counts)
top_sector = sector_counts.index[-1]
print("Top sector:", top_sector)

print("\n--- Q5: Highest revenue in top sector ---")
top_sector_df = companies[companies["sector"] == top_sector]
top_rev = top_sector_df.loc[top_sector_df["revenues"].idxmax()]
print(top_rev["company"], "| revenues:", top_rev["revenues"])

print("\n--- Q6: Top (country, sector) by company count ---")
country_sector = (
    companies.groupby(["country", "sector"])["company"]
    .count()
    .sort_values(ascending=False)
)
print(country_sector.head(5))

print("\n--- Q7: Companies that improved their rank ---")
improved = companies[companies["rank"] < companies["previous_rank"]]
print("Count of companies with improved rank:", len(improved))

print("\n--- Q8: Profit change rate & next-year profit estimate ---")
rate = companies["profit_change"] / companies["profits"]
companies["predicted_profit_vec"] = companies["profits"] * (1 + rate)
companies["predicted_profit_apply"] = companies.apply(
    lambda r: r["profits"] * (1 + (r["profit_change"] / r["profits"]))
    if pd.notna(r["profits"]) and r["profits"] != 0 and pd.notna(r["profit_change"])
    else pd.NA,
    axis=1,
)
print(companies[["company", "profits", "profit_change", "predicted_profit_vec", "predicted_profit_apply"]].head(10))
