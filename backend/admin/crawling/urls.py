from django.conf.urls import url

from admin.crawling import views

urlpatterns = {
    url(r'CrawlProcess', views.CrawlProcess),
    url(r'NewsProcess', views.NewsProcess),
}