"""
Retail Data Analysis Script
============================
Analyses fake_retail_data.csv using:
  - Descriptive statistics (numpy)
  - Sales trend analysis (monthly aggregation + numpy polyfit regression)
  - Category & region performance breakdowns
  - Discount impact analysis (t-test via numpy)
  - Customer age segmentation (numpy histogram / binning)
  - Anomaly / outlier detection (Z-score & IQR methods)
  - Correlation analysis (numpy corrcoef)
  - Matplotlib visualisations (saved to ./charts/)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
data = pd.read_csv(os.path.join(script_dir, "fake_retail_data.csv"))


# Convert Date column
data["Date"] = pd.to_datetime(data["Date"])

# Create charts folder
os.makedirs("charts", exist_ok=True)

print("\n===== INITIAL DATA INFO =====")
print(data.head())
print(data.info()) 

# -----------------------------
# DATA CLEANING
# -----------------------------

print("\n===== DATA CLEANING =====")

# Check missing values
print("\nMissing values before:")
print(data.isnull().sum())

# Drop rows with empty critical fields, we can't analyze without these
data = data.dropna(subset=["Date", "Sales"])

# Fill missing age with median
data["Customer_Age"] = data["Customer_Age"].fillna(data["Customer_Age"].median())

# Fill categorical missing values
data["Product_Category"] = data["Product_Category"].fillna("Unknown")
data["Region"] = data["Region"].fillna("Unknown")
data["Customer_Gender"] = data["Customer_Gender"].fillna("Unknown")

# Discount missing → False
data["Discount_Applied"] = data["Discount_Applied"].fillna(False)

print("\nMissing values after:")
print(data.isnull().sum())

# -----------------------------
# DESCRIPTIVE STATISTICS
# -----------------------------

print("\n===== DESCRIPTIVE STATISTICS =====")

sales = data["Sales"].values

# total of every sale
print("Total Sales:", np.sum(sales))
# arithmetic mean – the “average check”
print("Mean Sales:", np.mean(sales))
# median – the middle value after sorting, not pulled by extremes
print("Median Sales:", np.median(sales))
# standard deviation – how much the individual sales deviate from the mean
print("Std Dev:", np.std(sales))
print("Min:", np.min(sales))
print("Max:", np.max(sales))

# -----------------------------
# SALES TREND ANALYSIS
# -----------------------------

print("\n===== SALES TREND =====")

monthly_sales = data.groupby(data["Date"].dt.to_period("M"))["Sales"].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()

x = np.arange(len(monthly_sales))
y = monthly_sales.values

# Linear regression trend
coef = np.polyfit(x, y, 1)
trend = np.poly1d(coef)

plt.figure()
plt.plot(monthly_sales.index, y, label="Sales")
plt.plot(monthly_sales.index, trend(x), linestyle="--", label="Trend")
plt.title("Monthly Sales Trend")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/monthly_sales_trend.png")
plt.close()

# -----------------------------
# CATEGORY PERFORMANCE
# -----------------------------

print("\n===== CATEGORY PERFORMANCE =====")

category_sales = data.groupby("Product_Category")["Sales"].sum()
print(category_sales)

plt.figure()
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("charts/category_sales.png")
plt.close()

# -----------------------------
# REGION PERFORMANCE
# -----------------------------

print("\n===== REGION PERFORMANCE =====")

region_sales = data.groupby("Region")["Sales"].sum()
print(region_sales)

plt.figure()
region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("charts/region_sales.png")
plt.close()

# -----------------------------
# DISCOUNT IMPACT
# -----------------------------

print("\n===== DISCOUNT IMPACT =====")

discount_sales = data[data["Discount_Applied"] == True]["Sales"]
no_discount_sales = data[data["Discount_Applied"] == False]["Sales"]

print("Avg with discount:", np.mean(discount_sales))
print("Avg without discount:", np.mean(no_discount_sales))

# -----------------------------
# AGE SEGMENTATION
# -----------------------------

print("\n===== AGE SEGMENTATION =====")

ages = data["Customer_Age"].values

bins = [18,25,35,45,55,65,80]
hist, edges = np.histogram(ages, bins=bins)

print("Age segments:", hist)

plt.figure()
plt.hist(ages, bins=bins)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Customers")
plt.savefig("charts/age_distribution.png")
plt.close()

# -----------------------------
# OUTLIER DETECTION
# -----------------------------

print("\n===== OUTLIER DETECTION =====")

# Z-score method
z_scores = (sales - np.mean(sales)) / np.std(sales)
outliers_z = data[np.abs(z_scores) > 3]

print("\nZ-score outliers:")
print(outliers_z)

# IQR method
Q1 = np.percentile(sales, 25)
Q3 = np.percentile(sales, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = data[(sales < lower) | (sales > upper)]

print("\nIQR outliers:")
print(outliers_iqr)

# -----------------------------
# CORRELATION ANALYSIS
# -----------------------------

print("\n===== CORRELATION =====")

corr = np.corrcoef(data["Sales"], data["Customer_Age"])[0,1]

print("Correlation Sales vs Age:", corr)

# -----------------------------
# SALES DISTRIBUTION CHART
# -----------------------------

plt.figure()
plt.hist(sales, bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.savefig("charts/sales_distribution.png")
plt.close()

print("\nCharts saved in ./charts/")
print("Analysis complete.")