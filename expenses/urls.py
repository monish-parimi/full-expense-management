from django.urls import path
from . import views

urlpatterns = [
    path('add-expense/', views.add_expense, name='add_expense'),
    path('predict-next-month/', views.predict_next_month_view, name='predict_next_month'),
    path('get-expenses/', views.get_expenses, name='get_expenses'),
    path('delete-expense/<int:id>/', views.delete_expense, name='delete_expense'),
]
