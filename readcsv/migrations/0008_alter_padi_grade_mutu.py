# Generated by Django 3.2.23 on 2024-05-30 06:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('readcsv', '0007_alter_padi_grade_mutu'),
    ]

    operations = [
        migrations.AlterField(
            model_name='padi',
            name='grade_mutu',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]