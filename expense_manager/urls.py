from django.contrib import admin
from django.urls import path, include
from expenses import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('expenses.urls')),  # Connects to the expenses app
]
