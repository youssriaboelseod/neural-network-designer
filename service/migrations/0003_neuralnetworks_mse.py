# Generated by Django 3.0.8 on 2020-07-15 08:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0002_neuralnetworks_problem'),
    ]

    operations = [
        migrations.AddField(
            model_name='neuralnetworks',
            name='mse',
            field=models.IntegerField(default=-1),
        ),
    ]
