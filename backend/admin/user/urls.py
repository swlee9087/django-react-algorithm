from django.conf.urls import url
from admin.user import views


urlpatterns = [
    # 메인 urls.py 에서 매핑된 경로에 추가되어 들어가는 경로 http://127.0.0.1:8080/api/users/register
    url(r'', views.users, name='users'),
    url(r'/login', views.login),
    url(r'/<slug:id>', views.users),  # delete{id}
    url(r'', views.users),

]