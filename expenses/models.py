from django.db import models
from datetime import datetime

class Expense(models.Model):
    category = models.CharField(max_length=100)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField(default=datetime.now)
    category_type = models.CharField(max_length=50, choices=[('Income', 'Income'), ('Expenses', 'Expenses')])

    def __str__(self):
        return f'{self.category}: {self.amount}'

    @property
    def month(self):
        return self.date.strftime('%Y-%m')
