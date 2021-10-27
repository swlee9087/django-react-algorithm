from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from admin.crawling.models import Crawling, NewsCrawling

@api_view(['GET'])
@parser_classes([JSONParser])
def CrawlProcess(request):
    Crawling().process()
    return JsonResponse({'result': 'Crawling Success'})
@api_view(['GET'])
@parser_classes([JSONParser])
def NewsProcess(request):
    NewsCrawling().process()
    return JsonResponse({'result': 'Crawling Success'})
