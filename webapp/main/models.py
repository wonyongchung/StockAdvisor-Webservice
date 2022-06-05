from __future__ import unicode_literals
from django.db import models


class Main(models.Model):

    name = models.CharField(max_length=30)

    def __str__(self):
        # print(self.name + " | " + str(self.pk))
        return self.name + " | " + str(self.pk)

class Test(models.Model):

    name = models.CharField(max_length=50)

    def __str__(self):
        # print(self.name + " | " + str(self.pk))
        return self.name + " | " + str(self.pk)


"""
models.py에서 django.db.models 모듈에서 Model 상속해서 class 명 선언하면
db.sqlite3 폴더에 "상위폴더_class 이름" 형식으로 테이블이 생성됨
views 에서 해당 class을 가져와서 classname.objects.get(pk~) 하면 
해당 key 값으로 value 찾아서 반환
"""

