import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Path to store uploaded files and generated plots
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
METRICS_FILE = 'metrics.json'

# Ensure the upload and static folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def load_metrics():
    """Load existing metrics from a JSON file."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_metrics(metrics):
    """Save metrics to a JSON file."""
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        # Get the uploaded file and model type from the form
        file = request.files.get('file')
        model_type = request.form.get('model_type')
        months_ahead = int(request.form.get('months_ahead'))  # Get months_ahead from the form
        
        if not file or not model_type or not months_ahead:
            return "Please upload a CSV file, select a model, and specify months ahead.", 400

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, 'train.csv')
        file.save(file_path)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        if 'date' not in df.columns or 'sales' not in df.columns:
            return "The CSV file must contain 'date' and 'sales' columns.", 400

        # Convert 'date' to datetime, handle errors by setting to NaT
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with invalid dates
        df = df.dropna(subset=['date'])

        # Group by month and sum the sales
        df['month'] = df['date'].dt.to_period('M')
        df = df.groupby('month').agg({'sales': 'sum'}).reset_index()

        # Convert 'month' back to Timestamp for plotting
        df['date'] = df['month'].dt.to_timestamp()

        # Feature Engineering - Create lag features (previous 2 months' sales)
        df['lag_1'] = df['sales'].shift(1)
        df['lag_2'] = df['sales'].shift(2)
        df = df.dropna()  # Drop rows with NaN values

        # Prepare features and target
        X = df[['lag_1', 'lag_2']].values
        y = df['sales'].values

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select and initialize the model based on user input
        if model_type == 'LinearRegression':
            model = LinearRegression()
        elif model_type == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
        elif model_type == 'XGBoost':
            model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                booster='gbtree',
                tree_method='auto',
                random_state=42
            )
        else:
            return "Invalid model selected.", 400

        # Train the model
        model.fit(X_scaled, y)

        # Make predictions
        predictions = model.predict(X_scaled)

        # Evaluate the model
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)

        # Load existing metrics
        metrics = load_metrics()

        # Initialize metrics for the selected model if not present
        if model_type not in metrics:
            metrics[model_type] = {
                'mae': [],
                'mse': [],
                'rmse': [],
                'r2': []
            }

        # Append current metrics
        metrics[model_type]['mae'].append(mae)
        metrics[model_type]['mse'].append(mse)
        metrics[model_type]['rmse'].append(rmse)
        metrics[model_type]['r2'].append(r2)

        # Save updated metrics
        save_metrics(metrics)

        # Generate and save visualizations
        # 1. Actual vs Predicted Sales
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], y, label='Actual Sales', color='blue', linewidth=2)
        plt.plot(df['date'], predictions, label='Predicted Sales', color='orange', linestyle='--', linewidth=2)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Sales', fontsize=14)
        plt.title(f'Actual vs Predicted Sales ({model_type})', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        actual_vs_predicted_path = os.path.join(STATIC_FOLDER, 'actual_vs_predicted_sales.png')
        plt.savefig(actual_vs_predicted_path)
        plt.close()

        # 2. Future Sales Prediction (1 to 6 months ahead)
        future_sales = []
        for i in range(1, months_ahead + 1):
            latest_data = np.array([[df['lag_1'].iloc[-1], df['lag_2'].iloc[-1]]])
            latest_data_scaled = scaler.transform(latest_data)  # Scale the input features
            future_sales.append(model.predict(latest_data_scaled)[0])  # Add the prediction for this month
            df['lag_2'] = df['lag_1']  # Update lag_2 for next prediction
            df['lag_1'] = future_sales[-1]  # Update lag_1 for next prediction

        future_sales_plot_path = os.path.join(STATIC_FOLDER, 'future_sales.png')
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], y, label='Actual Sales', color='blue', linewidth=2)
        future_dates = pd.date_range(df['date'].iloc[-1], periods=months_ahead + 1, freq='M')[1:]
        plt.plot(future_dates, future_sales, marker='o', color='red', label=f'Predicted Future Sales ({months_ahead} Months)')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Sales', fontsize=14)
        plt.title(f'Predicted Future Sales ({months_ahead} Months)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(future_sales_plot_path)
        plt.close()

        # 3. Sales Trends Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], y, label='Sales', color='green', linewidth=2)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Sales', fontsize=14)
        plt.title('Sales Trend Over Time', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        sales_trends_path = os.path.join(STATIC_FOLDER, 'sales_trends.png')
        plt.savefig(sales_trends_path)
        plt.close()

        # 4. Model Performance Comparison (RMSE & R² score)
        models = ['LinearRegression', 'RandomForest', 'XGBoost']
        model_r2_scores = [metrics[model]['r2'][-1] if model in metrics else 0 for model in models]
        model_rmse_scores = [metrics[model]['rmse'][-1] if model in metrics else 0 for model in models]
        
        # Model comparison bar plot
        model_comparison_path = os.path.join(STATIC_FOLDER, 'model_comparison.png')
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(models, model_rmse_scores, color='skyblue', label='RMSE')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('RMSE', color='blue')
        
        ax2 = ax1.twinx()  # Create another axis for R² score
        ax2.plot(models, model_r2_scores, color='red', marker='o', label='R²', linestyle='-', linewidth=2)
        ax2.set_ylabel('R²', color='red')
        plt.title('Model Performance Comparison', fontsize=16)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.grid(alpha=0.3)
        plt.savefig(model_comparison_path)
        plt.close()

        # Render the results page with all plots
        return render_template(
            'train.html',
            mae=round(mae, 2),
            mse=round(mse, 2),
            rmse=round(rmse, 2),
            r2=round(r2, 2),
            actual_vs_predicted_img=actual_vs_predicted_path,
            future_sales_img=future_sales_plot_path,
            sales_trends_img=sales_trends_path,
            model_comparison_img=model_comparison_path
        )

    except Exception as e:
        return str(e)

# Reset metrics route
@app.route('/reset_metrics', methods=['POST'])
def reset_metrics():
    """Endpoint to reset stored metrics."""
    try:
        save_metrics({})
        return jsonify({"message": "Metrics reset successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to reset metrics: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
