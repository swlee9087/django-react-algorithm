from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from admin.myCV2.models import myCV2


@api_view(['GET'])
@parser_classes([JSONParser])
def lena(request):
    myCV2().lena()
    return JsonResponse({'lena': 'Success'})

@api_view(['GET'])
@parser_classes([JSONParser])
def girl(request):
    myCV2().girl()
    return JsonResponse({'girl': 'Success'})

@api_view(['GET'])
@parser_classes([JSONParser])
def face_detect(request):
    myCV2().face_detect()
    return JsonResponse({'Face Detection': 'Success'})

@api_view(['GET'])
@parser_classes([JSONParser])
def cat_mosaic(request):
    myCV2().cat_mosaic()
    return JsonResponse({'Cat Mosaic': 'Success'})

@api_view(['GET'])
@parser_classes([JSONParser])
def face_mosaic(request):
    myCV2().face_mosaic()
    return JsonResponse({'Face Mosaic': 'Success'})