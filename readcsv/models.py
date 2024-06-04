# from django.db import models
# import joblib
# # Create your models here.

# VARIETAS_CHOICES = (
#     (5, 'Pandan_Wangi'),
#     (3, 'Ciheran'),
#     (4, 'Mi_Kongga'),
#     (2, 'IR64'),
#     (1, 'Beras_Merah'),
#     (0, 'Beras_Hitam') 
#     )

# WARNA_CHOICES = (
#     (2, 'Merah'),
#     (3, 'Putih'),
#     (1, 'Hitam'),
#     (0, 'Coklat'),
#     )

# RASA_CHOICES = (
#     (0, 'Pulen'),
#     (1, 'Sangat_Pulen'),
#     )

# TEKNIK_CHOICES = (
#     (1, 'SRI'),
#     (0, 'Jajar_Legowo'),
#     )

# MUSIM__CHOICES = (
#     (0, 'Hujan'),
#     (1, 'Kemarau'),
#     )

# PENYAKIT_CHOICES = (
#     (0, 'Burung'),
#     (1, 'Penggerek_Batang'),
#     (3, 'Wereng_Coklat'),
#     (4, 'Wereng_Hijau'),
#     (2, 'Tikus'),
#     )

# class Padi(models.Model):
#     id = models.AutoField(primary_key=True)
#     varietas = models.PositiveIntegerField(choices=VARIETAS_CHOICES, null=True)
#     warna = models.PositiveIntegerField(choices=WARNA_CHOICES, null=True)
#     rasa = models.PositiveIntegerField(choices=RASA_CHOICES, null=True)
#     teknik = models.PositiveIntegerField(choices=TEKNIK_CHOICES, null=True)
#     musim = models.PositiveIntegerField(choices=MUSIM__CHOICES, null=True)
#     penyakit = models.PositiveIntegerField(choices=PENYAKIT_CHOICES, null=True)
#     ph = models.FloatField(null=True)
#     boron = models.FloatField(null=True)
#     fosfor = models.FloatField(null=True)
#     grade_mutu = models.PositiveIntegerField(blank=True,null=True)

#     # def save(self, *args, **kwargs):
#     #     model_path = 'data/best_model_m.pkl'
#     #     ml_model = joblib.load(model_path)
#     #     self.grade_mutu = ml_model.predict([[self.varietas, self.warna, self.rasa, self.teknik, self.musim, self.penyakit, self.ph, self.boron, self.fosfor]])[0]
#     #     return super().save(*args, **kwargs)

#     def __str__(self):
#         return str(self.id)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pickle
from django.http import HttpResponse

from django.shortcuts import render
from django.conf import settings
import pandas as pd
import os
