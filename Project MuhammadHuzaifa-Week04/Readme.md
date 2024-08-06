# Project: Analyzing Sales Data for a Retail Store
## Overview
This project involves analyzing sales data from a retail store. The main tasks include performing linear regression, data visualization, descriptive statistics, and probability distributions to understand and predict sales patterns.
## About the Author
Muhammad Huzaifa is a passionate data enthusiast and electrical engineering student with a keen interest in machine learning and data science. Connect with him on [GitHub](https://github.com/MHUZAIFAM) for more projects and collaborations.
## Additional Resources

- **Medium Article**: [Analyzing Sales Data for a Retail Store - A Step-by-Step Guide](https://medium.com/@mhuzaifa287e/analyzing-sales-data-for-a-retail-store-a-step-by-step-guide-1edac1024cd5)
- **GitHub Repository**: [Analyzing Sales Data for a Retail Store](https://github.com/MHUZAIFAM/100DaysOfBytewise/tree/main/Project%20MuhammadHuzaifa-Week04)

## Dataset
- **Name**: Retail Sales Dataset
- **File Path**: `C:/Users/pc/PycharmProjects/Project MuhammadHuzaifa-Week04/retail_sales_dataset.csv`

## Requirements
- Python libraries:
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

## Installation
To install the necessary libraries, run:

    pip install pandas matplotlib seaborn scikit-learn

## Steps Invloved

### Importing Libraries
Firstly we import the libraries installed earlier

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

### Loading the Dataset
Loading the dataset from the local machine

    url = 'C:/Users/pc/PycharmProjects/Project MuhammadHuzaifa-Week04/retail_sales_dataset.csv'
    sales_data = pd.read_csv(url)

### Displaying the First Few Rows of the Dataset

    print(sales_data.head())
    Descriptive Statistics
    python
    Copy code
    print(sales_data.describe())

### Visualizing the Distribution of Total Amounts
    plt.figure(figsize=(10, 6))
    sns.histplot(sales_data['Total Amount'], kde=True)
    plt.title('Total Amount Distribution')
    plt.xlabel('Total Amount')
    plt.ylabel('Frequency')
    plt.show()
    
### Correlation Heatmap for Numeric Columns
    numeric_columns = sales_data.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_columns.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
### Defining Features and Target Variable
    X = sales_data[['Quantity', 'Price per Unit']]
    y = sales_data['Total Amount']

### Splitting the Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
### Initializing and Training the Model
    model = LinearRegression()
    model.fit(X_train, y_train)
### Predicting on the Test Set
    y_pred = model.predict(X_test)
### Calculating Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
### Visualizing the Results
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Total Amount')
    plt.ylabel('Predicted Total Amount')
    plt.title('Actual vs Predicted Total Amount')
    plt.tight_layout()
    plt.show()
## Results
  - **Mean Squared Error (MSE)** : (output from the code)
  - **R-squared**: (output from the code)
## Visualizations
The project includes the following visualizations:

  - Distribution of Total Amount

    ![image](https://github.com/user-attachments/assets/5bfabe8f-869f-4789-a8c4-3161ab87e66a)

  - Correlation Heatmap
  
    ![image](https://github.com/user-attachments/assets/46a2dd27-6291-4589-bcc1-1d3d73f8d099)

      
  - Scatter plot of Actual vs Predicted Total Amount

    ![image](https://github.com/user-attachments/assets/f5ef0def-ddc0-49a3-a2b3-82e4ec2a6518)



## Conclusion
This project demonstrates the use of linear regression for predicting retail sales, along with various data visualization and statistical techniques to analyze and understand the sales data.

## Call to Action
Explore this project, try out the code, and delve deeper into machine learning for predictive analytics. Your feedback and comments are highly encouraged to foster learning and community engagement in data science.
