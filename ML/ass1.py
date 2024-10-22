import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def handle_missing_values_and_plot(file_path, column_to_compare):

    df = pd.read_csv(file_path)

    missing_values_count = df.isnull().sum()
    print("Initial Missing Values:\n", missing_values_count)

    numerical_columns = ['Model Year', 'Electric Range', 'Base MSRP','Legislative District','2020 Census Tract']  
    df_numeric = df[numerical_columns]
    mean_values = df_numeric.mean().round().astype(int)  
    valid_numeric_columns = mean_values[mean_values.notna()].index.tolist()

    print("Valid numeric columns with mean values (rounded):\n", mean_values[mean_values.notna()])
    df[valid_numeric_columns] = df[valid_numeric_columns].fillna(mean_values)

    '''
    # Step 3: Median Imputation
    df_median_imputed = df.fillna(df.median())

    # Step 4: Dropping Rows with Missing Values
    df_dropped_rows = df.dropna()

    # Step 5: Dropping Columns with Missing Values
    df_dropped_columns = df.dropna(axis=1)

    # Step 6: Mode Imputation (for categorical variables)
    for column in df.select_dtypes(include=['object']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

    # Step 7: Calculate and compare the mean of the specified column before and after imputation
    means = {
        'Original': df[column_to_compare].mean(),
        'Mean Imputed': df_mean_imputed[column_to_compare].mean(),
        'Median Imputed': df_median_imputed[column_to_compare].mean(),
        'Dropped Rows': df_dropped_rows[column_to_compare].mean(),
        'Dropped Columns': df_dropped_columns[column_to_compare].mean()
    }

    # Step 8: Plot the results for comparison
    plt.figure(figsize=(10, 5))
    plt.bar(means.keys(), means.values(), color='skyblue')
    plt.title(f'Mean Comparison for {column_to_compare} After Different Imputation Strategies')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    '''
# Example usage
file_path = 'Electric_Vehicle_Population_Data.csv'  # Update with your CSV file path
column_to_compare = 'Your_Column_Name'  # Replace with your actual column name
handle_missing_values_and_plot(file_path, column_to_compare)


'''
import pandas as pd

missing_values = ["n/a", "na", "--"]

df = pd.read_csv("Electric_Vehicle_Population_Data.csv", na_values=missing_values)


# Check for missing values and their frequency
missing_values_count = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values_count)
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)




'''


'''
# Step 4: Change column names with multiple words to abbreviations for plotting
abbreviated_columns = [''.join(word[0] for word in col.split()) if ' ' in col else col for col in df.columns]
missing_values_count.index = abbreviated_columns

# Step 5: Plotting the missing values as a bar chart with abbreviated column names
plt.figure(figsize=(10, 6))
missing_values_count.plot(kind='bar', color='skyblue')
plt.title('Frequency of Missing Values by Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Step 6: Show the plot
plt.tight_layout()  # Adjusts layout to fit labels
plt.show()


'''