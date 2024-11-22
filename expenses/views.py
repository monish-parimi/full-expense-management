import os
import sys
import django
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from .models import Expense
from django.db.models import Sum
from datetime import date, datetime, timedelta
import json
import logging
from decimal import Decimal
import joblib
from django.db.models import Sum

logger = logging.getLogger(__name__)

@csrf_exempt
def add_expense(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            category = data.get('category')
            amount = data.get('amount')
            date_str = data.get('date')
            category_type = data.get('category_type')  # Income or Expense
            
            if not category or not amount or not date_str or not category_type:
                return JsonResponse({'error': 'Missing required fields'}, status=400)

            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            expense = Expense(category=category, amount=amount, date=date, category_type=category_type)
            expense.save()
            return JsonResponse({'message': 'Expense added successfully!'})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        except Exception as e:
            logger.error(f"Error adding expense: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def get_expenses(request):
    date_str = request.GET.get('date')  # YYYY-MM-DD format
    month_str = request.GET.get('month')  # YYYY-MM format
    #category_type = request.GET.get('category_type')  # Optional filter by type
    
    try:
        if date_str:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            expenses = Expense.objects.filter(date=date)
            # if category_type:
            #     expenses = expenses.filter(category_type=category_type)
            total_expense_day = expenses.aggregate(total=Sum('amount'))['total'] or 0
            total_expense_month = Expense.objects.filter(
                date__year=date.year,
                date__month=date.month,
                #category_type=category_type if category_type else None
            ).aggregate(total=Sum('amount'))['total'] or 0
        elif month_str:
            year, month = map(int, month_str.split('-'))
            expenses = Expense.objects.filter(date__year=year, date__month=month)
            # if category_type:
            #     expenses = expenses.filter(category_type=category_type)
            total_expense_month = expenses.aggregate(total=Sum('amount'))['total'] or 0
            total_expense_day = None
        else:
            return JsonResponse({'error': 'Either date or month parameter is required'}, status=400)

        expenses_data = list(expenses.values('id', 'category', 'amount', 'date', 'category_type'))
        response_data = {
            'expenses': expenses_data,
            'total_expense_day': total_expense_day,
            'total_expense_month': total_expense_month
        }
        return JsonResponse(response_data)
    except ValueError:
        return JsonResponse({'error': 'Invalid date format'}, status=400)
    except Exception as e:
        logger.error(f"Error fetching expenses: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)



@csrf_exempt
def delete_expense(request, id):
    if request.method == 'DELETE':
        try:
            expense = Expense.objects.get(id=id)
            expense.delete()
            return JsonResponse({'message': 'Expense deleted successfully!'})
        except Expense.DoesNotExist:
            return JsonResponse({'error': 'Expense not found'}, status=404)
        except Exception as e:
            logger.error(f"Error deleting expense: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)



from expenses.train_model import train_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Directory and model file path
model_dir = os.path.join(PROJECT_ROOT, 'expense_manager', 'ml')
model_path = os.path.join(model_dir, 'expense_prediction_model.pkl')

def predict_next_month_view(request):
    today = date.today()

    # Get the current year and month
    current_year = today.year
    current_month = today.month

    # Calculate cumulative expenses up to the current month
    cumulative_expense = Expense.objects.filter(
        category_type="Expenses",  # Ensure this is the correct category
        date__lte=date(current_year, current_month, 1)  # Include up to current month
    ).aggregate(total=Sum('amount'))['total']

    # Handle the case where no data exists
    if not cumulative_expense:
        return JsonResponse({"error": "No expense data available for prediction."}, status=400)

    # Re-train the model with the latest data before making the prediction
    train_model()

    # Ensure the model file exists after training
    if not os.path.exists(model_path):
        return JsonResponse({"error": "Prediction model not found after retraining."}, status=500)

    # Load the prediction model
    model = joblib.load(model_path)

    # Prepare features for prediction (current year, current month, previous month's expense, and month-over-month change)
    features = [[current_year, current_month, float(cumulative_expense), 0]]  # Add 0 for the month-over-month change initially
    
    # Predict next month's expenses
    predicted_expense = model.predict(features)[0]

    # Calculate next month
    if current_month == 12:
        next_year = current_year + 1
        next_month = 1
    else:
        next_year = current_year
        next_month = current_month + 1

    # Return the response with prediction
    return JsonResponse({
        "current_year": current_year,
        "current_month": current_month,
        "next_year": next_year,
        "next_month": next_month,
        "cumulative_expense": float(cumulative_expense),
        "predicted_expense": round(predicted_expense, 2)
    })