# readcsv/models.py

from django.db import models

class ClassificationResult(models.Model):
    varietas = models.CharField(max_length=100)
    warna = models.CharField(max_length=100)
    rasa = models.CharField(max_length=100)
    musim = models.CharField(max_length=100)
    penyakit = models.CharField(max_length=100)
    teknik = models.CharField(max_length=100)
    ph = models.FloatField()
    boron = models.FloatField()
    fosfor = models.FloatField()
    prediction = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.varietas}, {self.warna}, {self.prediction}"
