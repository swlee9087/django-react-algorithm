from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from admin.tensor.models import Calculator, FashionClassification


@api_view(['GET'])
@parser_classes([JSONParser])
def process(request):
    Calculator().process()
    return JsonResponse({'result': 'Calculator Success'})

@api_view(['GET'])
@parser_classes([JSONParser])
def fashion(request):
    FashionClassification().fashion()
    return JsonResponse({'Fashion Classification': 'Success'})
