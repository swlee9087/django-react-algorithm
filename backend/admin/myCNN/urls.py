from django.conf.urls import url

from admin.myCNN import views

urlpatterns = {
    url(r'imdb_process', views.imdb_process),
}
