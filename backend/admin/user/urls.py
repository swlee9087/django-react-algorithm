from django.conf.urls import url
from admin.user import views
urlpatterns = [
    url(r'register', views.users),
    # 메인 urls.py 에서 매핑된 경로에 추가되어 들어가는 경로 http://127.0.0.1:8080/api/users/register
    url(r'list', views.users)
]