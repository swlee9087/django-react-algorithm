from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
from admin.housing.models import HousingService
from icecream import ic
import matplotlib.pyplot as plt


@api_view(['GET'])
@parser_classes([JSONParser])
def housing_info(request):
    HousingService().housing_info()
    return JsonResponse({'result': 'Housing Info Success'})


def housing_hist(request):
    HousingService().housing_hist()
    return JsonResponse({'result': 'Housing Hist SUCCESS'})


def income_cat_hist(request):
    hs = HousingService()
    hs.income_cat_hist()
    return JsonResponse({'result': 'income_cat_hist Save Success'})


def split_model_by_income_cat(request):
    hs = HousingService()
    hs.split_model()
    return JsonResponse({'result': 'income_cat_hist Save Success'})