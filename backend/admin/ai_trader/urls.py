from django.conf.urls import url

from admin.ai_trader import views

urlpatterns = {
    url(r'model_builder', views.model_builder),
}
