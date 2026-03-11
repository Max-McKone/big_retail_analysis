# Retail Analysis

## Project Overview

This Python project analyzes a sample retail dataset (`fake_retail_data.csv`) to provide insights into sales performance, customer behavior, and product trends. It performs:

- **Descriptive statistics** (total, mean, median, standard deviation of sales)
- **Sales trend analysis** (monthly aggregation with linear regression)
- **Category and region performance breakdowns**
- **Discount impact analysis**
- **Customer age segmentation**
- **Anomaly / outlier detection** (Z-score & IQR methods)
- **Correlation analysis** (Sales vs Customer Age)
- **Visualization of results** (charts saved to `./charts/`)

---

## Dataset

The CSV file should have the following columns:

| Column Name        | Description                                 |
|--------------------|---------------------------------------------|
| Date               | Date of purchase                            |
| Product_ID         | Unique product identifier                   |
| Product_Category   | Product category                            |
| Sales              | Sale amount                                 |
| Customer_Age       | Age of the customer                         |
| Customer_Gender    | Gender of the customer                      |
| Region             | Region of purchase                          |
| Discount_Applied   | Whether a discount was applied (True/False) |

