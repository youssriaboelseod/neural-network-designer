# Generated by Django 3.0.8 on 2020-07-15 11:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0004_auto_20200715_1100'),
    ]

    operations = [
        migrations.AlterField(
            model_name='neuralnetworks',
            name='mse',
            field=models.TextField(default='-1'),
        ),
    ]
