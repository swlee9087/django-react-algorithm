from django.conf.urls import url

from admin.tensor import views

urlpatterns = {
    url(r'process', views.process),
    url(r'fashion', views.fashion),
    # url(r'tf_function', views.tf_function),
    url(r'hook', views.hook),
}
