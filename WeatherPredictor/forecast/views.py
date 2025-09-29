from django.shortcuts import render

# Create your views here

from django.http import HttpResponse

def weather_view(request):
    return HttpResponse('<h1>Weather Prediction App</h1>')

