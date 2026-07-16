from django.urls import path
from . import views

urlpatterns = [
    # Single Page App
    path('', views.index, name='index'),
    
    # API Endpoints
    path('upload/', views.upload_file, name='upload_file'),
    path('ask/', views.ask_question, name='ask_question'),
    path('ask-general/', views.ask_general, name='ask_general'),
    path('clear-history/', views.clear_history, name='clear_history'),
]