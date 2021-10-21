from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from admin.iris.models import Iris


@api_view(['GET'])
@parser_classes([JSONParser])
def base(request):
    Iris().base()
    return JsonResponse({'Iris Base': 'Success'})\

@api_view(['GET'])
@parser_classes([JSONParser])
def advanced(request):
    Iris().advanced()
    return JsonResponse({'Iris Advanced': 'Success'})
