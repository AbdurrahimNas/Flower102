from django.db import models

# Create your models here.


class Flowers(models.Model):
    flower_image = models.FileField(null=False, blank=False)

    def __str__(self):
        return f"{self.id} {self.flower_image}"
    