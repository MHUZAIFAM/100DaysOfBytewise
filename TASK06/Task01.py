from sklearn.datasets import load_iris
import pandas as pd

#Exercise 1: Load a simple dataset (e.g., Iris dataset from Scikit-Learn) and print the first 5 rows
# Load the Iris dataset
iris = load_iris()

# Convert to a DataFrame for easier manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Print the first 5 rows
print(df.head())

#Exercise 2: Implement a function that takes a dataset and returns the number of features and samples
def dataset_info(data):
    num_samples, num_features = data.shape
    return num_features, num_samples

# Using the Iris dataset as an example
num_features, num_samples = dataset_info(df)
print(f"Number of features: {num_features}")
print(f"Number of samples: {num_samples}")

#Split a dataset into training and testing sets with an 80/20 split
from sklearn.model_selection import train_test_split

# Load target variable
target = iris.target

# Split dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

# Check the shapes of the split datasets
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

#Exercise 4: Explore the basic statistics of a dataset (mean, median, standard deviation)
# Basic statistics of the dataset
basic_stats = df.describe()
print(basic_stats)

#Exercise 5: Visualize the distribution of one of the features in the dataset using a histogram
import matplotlib.pyplot as plt

# Choose a feature to visualize (let's take the first one as an example)
feature_to_visualize = df.columns[0]

# Plotting histogram
plt.figure(figsize=(8, 6))
plt.hist(df[feature_to_visualize], bins=20, edgecolor='black')
plt.title(f'Distribution of {feature_to_visualize}')
plt.xlabel(feature_to_visualize)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
