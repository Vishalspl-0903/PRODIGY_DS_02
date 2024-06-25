import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'D:\Intern\Task2\iris\iris.data'  
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
iris_df = pd.read_csv(file_path, header=None, names=column_names)

iris_df['target'] = iris_df['target'].astype('category').cat.codes

print("First few rows of the dataset:")
print(iris_df.head())

missing_values = iris_df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

data_types = iris_df.dtypes
print("\nData types of each column:")
print(data_types)

summary_stats = iris_df.describe()
print("\nSummary statistics of the dataset:")
print(summary_stats)

print("\nHistograms for each feature:")
iris_df.hist(figsize=(10, 8))
plt.show()

print("\nBox plots for each feature:")
plt.figure(figsize=(10, 8))
sns.boxplot(data=iris_df)
plt.show()

print("\nPairplot to show pairwise relationships:")
sns.pairplot(iris_df, hue='target', markers=["o", "s", "D"])
plt.show()

print("\nCorrelation heatmap:")
plt.figure(figsize=(10, 8))
sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm')
plt.show()

print("\nViolin plots for each feature by target class:")
plt.figure(figsize=(10, 8))
for column in iris_df.columns[:-1]:
    sns.violinplot(x='target', y=column, data=iris_df)
    plt.title(column)
    plt.show()
