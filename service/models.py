from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.db.models import CASCADE, ImageField


class NeuralNetworks(models.Model):
    id = models.AutoField(primary_key=True)
    creator = models.TextField(null=False)
    problem = models.TextField(default='')
    mse = models.TextField(default='-1')
    neuron_list = ArrayField(models.IntegerField(null=True, blank=True), null=True, blank=True)
    activation_list = ArrayField(models.CharField(max_length=1000, null=True, blank=True), null=True, blank=True)

    def __repr__(self):
        return str(self.neuron_list) + str(self.activation_list)

    def __str__(self):
        return str(self.neuron_list) + str(self.activation_list)


class Graphs(models.Model):
    id = models.AutoField(primary_key=True)
    nnb_id = models.ForeignKey(NeuralNetworks, on_delete=CASCADE)
    xplot = ArrayField(models.FloatField(null=True, blank=True), null=True, blank=True)
    yplot = ArrayField(models.FloatField(null=True, blank=True), null=True, blank=True)
    yhatplot = ArrayField(models.FloatField(null=True, blank=True), null=True, blank=True)
