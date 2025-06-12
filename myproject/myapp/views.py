# views.py
from django.http import JsonResponse
from .models import Employee

def employee_list(request):
    employees = Employee.objects.all().values('name', 'age', 'department')
    return JsonResponse(list(employees), safe=False)