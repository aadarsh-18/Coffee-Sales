# coffee_sales_project.py

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Generate synthetic coffee sales data
def generate_data(start_date='2023-01-01', end_date='2024-01-01'):
    logging.info("Generating synthetic coffee sales data...")
    dates = pd.date_range(start_date, end_date)
    np.random.seed(42)
    data = pd.DataFrame({'Date': dates})
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['Promotion'] = np.random.binomial(1, 0.3, len(data))  # 30% days have promotion
    data['Holiday'] = np.where(data['Date'].dt.dayofweek >= 5, 1, 0)  # Weekend as holiday
    
    # Base sales
    base_sales = 100 + (data['Month'] - 6).abs() * 5  # Higher sales mid-year
    
    # Sales influenced by promotion (+20%), holiday (-30%), randomness
    data['Sales'] = (base_sales * (1 + 0.2 * data['Promotion']) * (1 - 0.3 * data['Holiday'])
                     + np.random.normal(0, 5, len(data))).round().astype(int)
    data['Sales'] = data['Sales'].apply(lambda x: max(x, 0))  # No negative sales
    logging.info("Data generation complete.")
    return data

# Step 2: Store data to SQLite
def store_to_sql(data, db_name='coffee_sales.db'):
    logging.info("Storing data to SQLite database...")
    conn = sqlite3.connect(db_name)
    data.to_sql('sales', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    logging.info("Data stored successfully.")

# Step 3: Load data from SQL
def load_from_sql(db_name='coffee_sales.db'):
    logging.info("Loading data from SQLite database...")
    conn = sqlite3.connect(db_name)
    df = pd.read_sql('SELECT * FROM sales', conn)
    conn.close()
    logging.info("Data loaded successfully.")
    return df

# Step 4: EDA visualization
def plot_eda(df):
    logging.info("Plotting sales trend...")
    plt.figure(figsize=(14,6))
    sns.lineplot(x='Date', y='Sales', data=df)
    plt.title('Coffee Sales Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    sns.boxplot(x='Promotion', y='Sales', data=df)
    plt.title('Sales Distribution: Promotion vs No Promotion')
    plt.show()

# Step 5: Feature engineering
def create_features(df):
    logging.info("Creating features for ML model...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df['Lag_1'] = df['Sales'].shift(1)  # Sales previous day
    df['Lag_7'] = df['Sales'].shift(7)  # Sales previous week
    df = df.dropna()
    return df

# Step 6: Train ML model
def train_model(df):
    logging.info("Training Random Forest Regressor model...")
    features = ['Promotion', 'Holiday', 'Day', 'Weekday', 'Lag_1', 'Lag_7']
    target = 'Sales'
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"MAE: {mae:.2f}")

    # Add predictions to test set for review
    test_results = X_test.copy()
    test_results['Actual'] = y_test
    test_results['Predicted'] = preds.round().astype(int)

    return model, test_results

# Step 7: Export results to Excel
def export_to_excel(df, filename='Coffee_Sales_Report.xlsx'):
    logging.info(f"Exporting results to {filename}...")
    df.to_excel(filename, index=False)
    logging.info("Export complete.")

# Full pipeline execution
def run_pipeline():
    data = generate_data()
    store_to_sql(data)
    df = load_from_sql()
    plot_eda(df)
    df_features = create_features(df)
    model, test_results = train_model(df_features)
    export_to_excel(test_results)

if __name__ == "__main__":
    run_pipeline()
