import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv(r"C:\Users\hp\Downloads\smartphone dataset\Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

print("First 5 Rows:\n", df.head())
print("\nColumns:\n", df.columns)

# Handle missing values
df = df.dropna()


# 1. Age Distribution

plt.figure(figsize=(8,5))
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution of Users")
plt.xlabel("Age")
plt.show()


# 2. Daily Screen Time Distribution

plt.figure(figsize=(8,5))
sns.histplot(df['daily_screen_time_hours'], kde=True)
plt.title("Daily Screen Time Distribution")
plt.xlabel("Hours")
plt.show()


# 3. Addiction Level Count

plt.figure(figsize=(6,4))
sns.countplot(x='addiction_level', data=df)
plt.title("Addiction Level Distribution")
plt.show()


# 4. Screen Time vs Addiction Level (VERY IMPORTANT)

plt.figure(figsize=(8,6))
sns.boxplot(x='addiction_level', y='daily_screen_time_hours', data=df)
plt.title("Screen Time vs Addiction Level")
plt.show()


# 5. Social Media vs Addiction

plt.figure(figsize=(8,6))
sns.boxplot(x='addiction_level', y='social_media_hours', data=df)
plt.title("Social Media Usage vs Addiction Level")
plt.show()


# 6. Gaming Hours vs Addiction

plt.figure(figsize=(8,6))
sns.boxplot(x='addiction_level', y='gaming_hours', data=df)
plt.title("Gaming Hours vs Addiction Level")
plt.show()


# 7. Sleep vs Screen Time

plt.figure(figsize=(8,6))
sns.scatterplot(x='daily_screen_time_hours', y='sleep_hours', hue='addiction_level', data=df)
plt.title("Screen Time vs Sleep Hours")
plt.show()


# 8. Notifications vs Addiction

plt.figure(figsize=(8,6))
sns.boxplot(x='addiction_level', y='notifications_per_day', data=df)
plt.title("Notifications vs Addiction Level")
plt.show()


# 9. Correlation Heatmap (VERY IMPORTANT)

plt.figure(figsize=(10,6))
corr = df.select_dtypes(include=['int64','float64']).corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 10. Gender Distribution (Keep only 1 pie chart)

gender_counts = df['gender'].value_counts()

plt.figure()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.show()


# Statistical Summary

print("\nSummary Statistics:\n", df.describe())

# Mean & Std Example
print("\nAverage Screen Time:", np.mean(df['daily_screen_time_hours']))
print("Max Screen Time:", np.max(df['daily_screen_time_hours']))
print("Standard Deviation:", np.std(df['daily_screen_time_hours']))

print("\nProject Completed Successfully")
