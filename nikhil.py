import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:\\Users\\ajayk\\Desktop\\Candy_Sales.csv")
sns.set(style="whitegrid")

# Convert date columns
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# -----------------------
# 1. Sales by Region
# -----------------------
plt.figure(figsize=(10, 6))
sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
sns.barplot(x=sales_by_region.index, y=sales_by_region.values, palette='YlGnBu')
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------
# 2. Most and Least Profitable Products
# -----------------------
profit_by_product = df.groupby('Product Name')['Gross Profit'].sum().sort_values(ascending=False)

top5 = profit_by_product.head(5)
bottom5 = profit_by_product.tail(5)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=top5.values, y=top5.index, palette="summer", ax=axes[0])
axes[0].set_title("Top 5 Most Profitable Products")
axes[0].set_xlabel("Gross Profit")

sns.barplot(x=bottom5.values, y=bottom5.index, palette="autumn", ax=axes[1])
axes[1].set_title("Top 5 Least Profitable Products")
axes[1].set_xlabel("Gross Profit")

plt.tight_layout()
plt.show()

# -----------------------
# 3. Customer Purchase Patterns
# -----------------------

# Orders per customer
orders_per_customer = df.groupby('Customer ID')['Order ID'].nunique()

# Average spend per customer
avg_spend = df.groupby('Customer ID')['Sales'].mean()

# Plot
plt.figure(figsize=(12, 5))
sns.histplot(orders_per_customer, bins=30, kde=True, color='plum')
plt.title("Distribution of Orders per Customer")
plt.xlabel("Number of Orders")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(12, 5))
sns.histplot(avg_spend, bins=30, kde=True, color='lightseagreen')
plt.title("Average Spend per Customer")
plt.xlabel("Sales Amount")
plt.ylabel("Number of Customers")
plt.show()

# -----------------------
# 4. Shipping Modes Impact
# -----------------------

# Average delivery time
df['Delivery Time'] = (df['Ship Date'] - df['Order Date']).dt.days
delivery_by_mode = df.groupby('Ship Mode')['Delivery Time'].mean()
sales_by_mode = df.groupby('Ship Mode')['Sales'].sum()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x=delivery_by_mode.index, y=delivery_by_mode.values, ax=ax[0], palette="cool")
ax[0].set_title("Average Delivery Time by Ship Mode")
ax[0].set_ylabel("Days")

sns.barplot(x=sales_by_mode.index, y=sales_by_mode.values, ax=ax[1], palette="rocket")
ax[1].set_title("Total Sales by Ship Mode")
ax[1].set_ylabel("Sales")

plt.tight_layout()
plt.show()

# -----------------------
# 5. Top Selling Products (by Sales and Units)
# -----------------------
top_sales = df.groupby('Product Name')[['Sales', 'Units']].sum().sort_values(by='Sales', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_sales['Sales'], y=top_sales.index,palette="Spectral")
plt.title("Top 10 Products by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product")
plt.show()

plt.figure(figsize=(12, 6))
top_units = top_sales.sort_values(by='Units', ascending=False)
sns.barplot(x=top_units['Units'], y=top_units.index, palette="viridis")
plt.title("Top 10 Products by Units Sold")
plt.xlabel("Units Sold")
plt.ylabel("Product")
plt.show()
# -----------------------
# 6. Pair Plot (for numeric variables)
# -----------------------
numeric_cols = ['Sales', 'Units', 'Gross Profit', 'Delivery Time']
# sns.pairplot(df[numeric_cols], diag_kind='kde', palette="Set2")
sns.pairplot(df, vars=numeric_cols, hue='Region', diag_kind='kde', palette="Set2")


plt.suptitle("Pair Plot of Key Numeric Variables", y=1.02)
plt.show()

# -----------------------
# 7. Correlation Heatmap
# -----------------------
plt.figure(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()
