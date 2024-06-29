#Exercise10: Implement a simple linear regression model using Scikit-Learn and print the model coefficients
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset_iris.csv')

# Prepare the data for linear regression (example: using sepal length 'sepal length (cm)' as input X)
X = df[['sepal length (cm)']]
y = df['species']  # Using 'species' as the target variable for demonstration

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Print the coefficients
print("Linear Regression Coefficients:")
print(f"Coefficient (slope): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Predictions
y_pred = model.predict(X_test)

# Plotting the linear regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression Model')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Species')
plt.title('Linear Regression Model on Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()
