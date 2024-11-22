import os
import sys
import django
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Django setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "expense_manager.settings")
django.setup()

from expenses.models import Expense

model_dir = os.path.join(PROJECT_ROOT, 'expense_manager', 'ml')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'expense_prediction_model.pkl')


# def load_data():
#     expenses = Expense.objects.filter(category_type="Expenses").values('amount', 'date')
#     df = pd.DataFrame(list(expenses))
#     if df.empty:
#         raise ValueError("No expense data found in the database. Ensure the Expense table has records.")
    
#     df['date'] = pd.to_datetime(df['date'])
#     df['Year'] = df['date'].dt.year
#     df['Month'] = df['date'].dt.month
#     print("Data before grouping:", df.head())
    
#     grouped = df.groupby(['Year', 'Month'], as_index=False).agg({'amount': 'sum'})
#     grouped['Cumulative_Expense'] = grouped['amount'].cumsum()
#     print("Grouped DataFrame:", grouped.head())

#     return grouped



# def train_model():
#     data = load_data()
#     X = data[['Year', 'Month', 'Cumulative_Expense']]
#     y = data['amount']

#     # Split data into training and testing
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Evaluate model
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Model trained with Mean Squared Error: {mse}")

#     # Save the model
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")


def load_data():
    expenses = Expense.objects.filter(category_type="Expenses").values('amount', 'date')
    df = pd.DataFrame(list(expenses))
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Previous_Month_Expense'] = df['amount'].shift(1)
    df['Month_Over_Month_Change'] = df['amount'].pct_change()
    
    df.dropna(inplace=True)

    print("Data before grouping:", df.head())
    
    grouped = df.groupby(['Year', 'Month'], as_index=False).agg({'amount': 'sum'})
    grouped['Cumulative_Expense'] = grouped['amount'].cumsum()
    print("Grouped DataFrame:", grouped.head())

    return df[['Year', 'Month', 'Previous_Month_Expense', 'Month_Over_Month_Change', 'amount']]
    return grouped


def train_model():
    data = load_data()
    X = data[['Year', 'Month', 'Previous_Month_Expense', 'Month_Over_Month_Change']]
    y = data['amount']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained with Mean Squared Error: {mse}")

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()
