from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from admin.myRNN.models import myRNN


@api_view(['GET'])
@parser_classes([JSONParser])
def ram_price(request):
    myRNN().ram_price()
    return JsonResponse({'RNN ram_price': 'Success'})

@api_view(['GET'])
@parser_classes([JSONParser])
def kia_predict(request):
    myRNN().kia_predict()
    return JsonResponse({'RNN kia_predict': 'Success'})