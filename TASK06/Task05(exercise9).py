#Exercise 9: Load a CSV file into a Pandas DataFrame and print summary statistics for each column.
#Creating a csv file first
from sklearn.datasets import load_iris
import pandas as pd
# Load the Iris dataset from Scikit-Learn
iris = load_iris()
# Convert to a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the target variable 'species' to the DataFrame
df['species'] = iris.target
# Save the DataFrame to a CSV file
df.to_csv('dataset_iris.csv', index=False)
print("CSV file 'dataset_iris.csv' has been created successfully.")

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset_iris.csv')

# Print the first 5 rows
print("First 5 rows of the DataFrame:")
print(df.head())

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset_iris.csv')

# Print the first 5 rows
print("First 5 rows of the DataFrame:")
print(df.head())
