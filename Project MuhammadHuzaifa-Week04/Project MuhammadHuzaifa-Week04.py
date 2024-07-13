#Project: Analyzing Sales Data for a Retail Store
#Dataset: Retail Sales Dataset
#Requirnment: Linear regression, data visualization, descriptive statistics, probability distributions

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = 'C:/Users/pc/PycharmProjects/Project MuhammadHuzaifa-Week04/retail_sales_dataset.csv'
sales_data = pd.read_csv(url)

# Display the first few rows of the dataset
print(sales_data.head())

# Descriptive statistics
print(sales_data.describe())

# Visualize the distribution of total amounts
plt.figure(figsize=(10, 6))
sns.histplot(sales_data['Total Amount'], kde=True)
plt.title('Total Amount Distribution')
plt.xlabel('Total Amount')
plt.ylabel('Frequency')

# Correlation heatmap for numeric columns
numeric_columns = sales_data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_columns.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')

# Define features and target variable
# Here, we are assuming that Quantity, Price per Unit, and Total Amount could be important features.
X = sales_data[['Quantity', 'Price per Unit']]
y = sales_data['Total Amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Total Amount')
plt.ylabel('Predicted Total Amount')
plt.title('Actual vs Predicted Total Amount')

# Show all plots
plt.tight_layout()
plt.show()
