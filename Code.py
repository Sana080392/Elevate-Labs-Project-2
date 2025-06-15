import pandas as pd

# 1. Load the updated dataset
file_path = "C:/Users/sanaf/Desktop/EL Project 2/Shark Tank India.csv"
df = pd.read_csv(file_path)

# 2. Convert 'Pitchers Average Age' from label to number
age_mapping = {
    'Young': 25,
    'Middle': 40,
    'Old': 55
}
df['Pitchers Average Age'] = df['Pitchers Average Age'].map(age_mapping)

# 3. Fill missing values
df.fillna({
    'Total Deal Amount': 0,
    'Total Deal Equity': 0,
    'Accepted Offer': 0,
    'Pitchers Average Age': df['Pitchers Average Age'].median(),
    'Industry': 'Unknown'
}, inplace=True)

# 4. Convert money columns to Crores
money_columns = ['Total Deal Amount', 'Valuation Requested', 'Deal Valuation']
for col in money_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col + ' (Crores)'] = df[col] / 100

# 5. Standardize industry names
df['Industry'] = df['Industry'].astype(str).str.lower().str.strip()

# 6. Create Deal Success field
df['Deal Success'] = df['Accepted Offer'].apply(lambda x: 1 if x == 1 else 0)

# 7. Investor Count
df['Investor Count'] = pd.to_numeric(df['Number of Sharks in Deal'], errors='coerce').fillna(0).astype(int)

# 8. Equity per Lakh
df['Total Deal Amount'] = pd.to_numeric(df['Total Deal Amount'], errors='coerce')
df['Total Deal Equity'] = pd.to_numeric(df['Total Deal Equity'], errors='coerce')

df['Equity per Lakh'] = df.apply(
    lambda row: row['Total Deal Amount'] / row['Total Deal Equity']
    if pd.notnull(row['Total Deal Equity']) and row['Total Deal Equity'] != 0 else None,
    axis=1
)

# 9. Age group (based on converted age)
def categorize_age(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 30:
        return '<30'
    elif 30 <= age < 50:
        return '30-50'
    else:
        return '>50'

df['Pitcher Age Group'] = df['Pitchers Average Age'].apply(categorize_age)

# 10. Save cleaned dataset
cleaned_path = "C:/Users/sanaf/Desktop/EL Project 2/Shark Tank India Cleaned.csv"
df.to_csv(cleaned_path, index=False)

print("✅ Data cleaned and saved to:")
print(cleaned_path)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = r"C:\Users\sanaf\Desktop\EL Project 2\Shark Tank India Cleaned.csv"
df = pd.read_csv(file_path)

# Optional: Quick view of all columns
print("Columns in dataset:", df.columns.tolist())

# Set visual style
sns.set(style='whitegrid')

# ---------------------------
# 1. Pitcher Age Group vs Deal Success
# ---------------------------
df['Pitcher Age Group'] = pd.cut(df['Pitchers Average Age'], bins=[15, 25, 35, 45, 55], 
                                 labels=['18–25', '26–35', '36–45', '46–55'])

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='Pitcher Age Group', y='Accepted Offer', estimator='mean', hue='Pitcher Age Group', palette='coolwarm', legend=False)
plt.title('Deal Success Rate by Pitcher Age Group')
plt.ylabel('Success Rate')
plt.xlabel('Pitcher Age Group')
plt.tight_layout()
plt.show()

# ---------------------------
# 2. Industry vs Deal Success (Top 10)
# ---------------------------
top_industries = df['Industry'].value_counts().nlargest(10).index
df_top = df[df['Industry'].isin(top_industries)]

plt.figure(figsize=(10, 5))
sns.barplot(data=df_top, x='Industry', y='Accepted Offer', estimator='mean', hue='Industry', palette='viridis', legend=False)
plt.title('Deal Success Rate by Top 10 Industries')
plt.ylabel('Success Rate')
plt.xlabel('Industry')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ---------------------------
# 3. Investor Count vs Total Deal Amount
# ---------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Number of Sharks in Deal', y='Total Deal Amount', hue='Number of Sharks in Deal', palette='Set2', legend=False)
plt.title('Total Deal Amount by Number of Sharks')
plt.ylabel('Amount (₹)')
plt.xlabel('Investor Count')
plt.tight_layout()
plt.show()

# ---------------------------
# 4. Top 10 States by Number of Startups
# ---------------------------
plt.figure(figsize=(10, 5))
top_states = df['Pitchers State'].value_counts().nlargest(10)
sns.barplot(x=top_states.index, y=top_states.values, palette='crest')
plt.title('Top 10 States by Number of Startups')
plt.ylabel('Number of Pitches')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# 5. Gender-wise Deal Success (Derived Gender Column)
# ---------------------------
df['Gender'] = df.apply(
    lambda row: 'Male' if row['Male Presenters'] > row['Female Presenters']
    else 'Female' if row['Female Presenters'] > row['Male Presenters']
    else 'Mixed', axis=1)

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='Gender', y='Accepted Offer', hue='Gender', palette='Set1', legend=False)
plt.title('Deal Success Rate by Gender Group')
plt.ylabel('Success Rate')
plt.xlabel('Gender')
plt.tight_layout()
plt.show()

# ---------------------------
# 6. Valuation vs Deal Amount
# ---------------------------
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='Valuation Requested', y='Total Deal Amount', hue='Accepted Offer', palette='coolwarm')
plt.title('Valuation vs Deal Amount')
plt.xlabel('Valuation Requested (₹ Crores)')
plt.ylabel('Total Deal Amount (₹ Lakhs)')
plt.tight_layout()
plt.show()

# ---------------------------
# 7. Shark Investment Distribution
# ---------------------------
shark_cols = ['Namita Investment Amount', 'Vineeta Investment Amount', 'Anupam Investment Amount',
              'Aman Investment Amount', 'Peyush Investment Amount', 'Ritesh Investment Amount', 'Amit Investment Amount']

df_sharks = df[shark_cols].fillna(0)
shark_investments = df_sharks.sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=shark_investments.index, y=shark_investments.values, palette='magma')
plt.title('Total Investment Amount by Shark')
plt.ylabel('Investment Amount (₹)')
plt.xlabel('Shark')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# BONUS: Correlation Heatmap
# ---------------------------
plt.figure(figsize=(12, 6))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('Correlation Matrix (Numerical Columns)')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv("C:/Users/sanaf/Desktop/EL Project 2/Shark Tank India Cleaned.csv")

print("="*60)
print("🔍 Basic Info")
print("="*60)
print(df.info())

print("\n" + "="*60)
print("📊 Summary Statistics (Numerical Columns)")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("🧩 Missing Values")
print("="*60)
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing.sort_values(ascending=False))

print("\n" + "="*60)
print("🔢 Unique Values per Column")
print("="*60)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n" + "="*60)
print("💡 Top Industries Pitched")
print("="*60)
print(df['Industry'].value_counts().head(10))

print("\n" + "="*60)
print("🗺️ Top 10 States (by Pitch Count)")
print("="*60)
if 'Pitchers State' in df.columns:
    print(df['Pitchers State'].value_counts().head(10))
else:
    print("Column 'Pitchers State' not found.")

print("\n" + "="*60)
print("💰 Most Asked Amounts")
print("="*60)
print(df['Original Ask Amount'].value_counts().head(10))

print("\n" + "="*60)
print("✅ Deal Success Rate")
print("="*60)
deal_success = df['Accepted Offer'].value_counts()
print(deal_success)

print("\n" + "="*60)
print("🤝 Average Deal Amount by Number of Sharks")
print("="*60)
if 'Number of Sharks in Deal' in df.columns:
    print(df.groupby('Number of Sharks in Deal')['Total Deal Amount'].mean())
else:
    print("Column 'Number of Sharks in Deal' not found.")

# ============ Visualizations ============

# Set style
sns.set(style="whitegrid")

# 1. Heatmap of Correlation
plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 2. Distribution of Deal Amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['Total Deal Amount'], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Total Deal Amounts")
plt.xlabel("Total Deal Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3. Number of Pitches per Industry
plt.figure(figsize=(12, 6))
industry_counts = df['Industry'].value_counts().head(10)
sns.barplot(x=industry_counts.index, y=industry_counts.values, palette="viridis")
plt.title("Top 10 Industries by Pitch Count")
plt.xticks(rotation=45)
plt.ylabel("Number of Pitches")
plt.tight_layout()
plt.show()

# 4. Deal Success Rate by Industry
plt.figure(figsize=(12, 6))
df['Deal Success'] = df['Accepted Offer'].apply(lambda x: 1 if x else 0)
industry_success = df.groupby('Industry')['Deal Success'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=industry_success.index, y=industry_success.values, palette="coolwarm")
plt.title("Average Deal Success Rate by Industry")
plt.xticks(rotation=45)
plt.ylabel("Success Rate")
plt.tight_layout()
plt.show()

# 5. Boxplot of Valuations
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='Valuation Requested', palette="Set3")
plt.title("Boxplot of Valuation Requested")
plt.tight_layout()
plt.show()

# 6. Pairplot of Key Financials
plt.figure(figsize=(12, 8))
sns.pairplot(df[['Yearly Revenue', 'Monthly Sales', 'Gross Margin', 'Net Margin']], kind='scatter')
plt.suptitle("Pairplot of Revenue and Margins", y=1.02)
plt.tight_layout()
plt.show()

# 7. Countplot of Pitcher Age Groups
plt.figure(figsize=(8, 5))
if 'Pitcher Age Group' in df.columns:
    sns.countplot(data=df, x='Pitcher Age Group', palette='pastel')
    plt.title("Pitcher Age Group Distribution")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Pitcher Age Group' not found.")

print("\n✅ EDA Completed Successfully!")

import pandas as pd
import io

# Load your dataset
df = pd.read_csv("C:/Users/sanaf/Desktop/EL Project 2/Shark Tank India Cleaned.csv")

# Capture df.info() into a string
info_buffer = io.StringIO()
df.info(buf=info_buffer)
info_text = info_buffer.getvalue()

# Prepare other EDA outputs
describe = df.describe()
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]

unique_values = pd.DataFrame({
    'Column': df.columns,
    'Unique Values': [df[col].nunique() for col in df.columns]
})

industry_counts = df['Industry'].value_counts().head(10)
ask_amounts = df['Original Ask Amount'].value_counts().head(10)
deal_success = df['Accepted Offer'].value_counts()
sharks_vs_deals = df.groupby('Number of Sharks in Deal')['Total Deal Amount'].mean()

if 'Pitchers State' in df.columns:
    top_states = df['Pitchers State'].value_counts().head(10)
else:
    top_states = pd.Series(["Column not found"], index=["Message"])

# Deal success rate by industry
df['Deal Success'] = df['Accepted Offer'].apply(lambda x: 1 if x else 0)
industry_success = df.groupby('Industry')['Deal Success'].mean().sort_values(ascending=False).head(10)

# Save all to Excel
output_path = "C:/Users/sanaf/Desktop/EL Project 2/EDA_Analysis_Output.xlsx"
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    pd.DataFrame({"Info": [info_text]}).to_excel(writer, sheet_name="Info", index=False)
    describe.to_excel(writer, sheet_name="Summary Stats")
    missing.to_excel(writer, sheet_name="Missing Values")
    unique_values.to_excel(writer, sheet_name="Unique Values", index=False)
    industry_counts.to_frame("Pitch Count").to_excel(writer, sheet_name="Top Industries")
    ask_amounts.to_frame("Count").to_excel(writer, sheet_name="Top Ask Amounts")
    deal_success.to_frame("Count").to_excel(writer, sheet_name="Deal Success Rate")
    sharks_vs_deals.to_frame("Avg Deal Amount").to_excel(writer, sheet_name="Sharks vs Deals")
    top_states.to_frame("Pitch Count").to_excel(writer, sheet_name="Top States")
    industry_success.to_frame("Success Rate").to_excel(writer, sheet_name="Success by Industry")

print("✅ All EDA outputs saved to Excel successfully!")
