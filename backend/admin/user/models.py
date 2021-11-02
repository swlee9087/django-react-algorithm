from django.db import models


class User(models.Model):
    use_in_migrations = True
    username = models.CharField(primary_key=True, max_length=15)
    password = models.CharField(max_length=10)
    name = models.TextField()
    email = models.EmailField()
    birth = models.TextField()
    address = models.TextField()

    def __str__(self):
        return f'[{self.pk}] {self.username}'

    class Meta:
        db_table = "users"
