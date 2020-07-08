import copy

from backend.neural_network_config import *


def mutate_layer_neurons(neurons):
    """
    Mutation by changing quantity of neurons in random layer
    :param neurons:
    :return:
    """
    new_neurons = copy.deepcopy(neurons)
    ind = random.randint(0, len(neurons) - 1)
    new_neurons[ind] = get_random_neurons()
    return new_neurons


def mutate_layer_activation(acts):
    """
    Mutation by changing activation function in random layer
    :param acts:
    :return:
    """
    new_acts = copy.deepcopy(acts)
    ind = random.randint(0, len(acts) - 1)
    new_acts[ind] = get_activation_random()
    return new_acts


def delete_random_layer(neurons, acts):
    """
    Mutation by deleting random layer
    :param neurons:
    :param acts:
    :return:
    """
    new_neurons = copy.deepcopy(neurons)
    new_acts = copy.deepcopy(acts)
    ind = random.randint(0, len(acts) - 1)
    del new_neurons[ind]
    del new_acts[ind]
    return new_neurons, new_acts


def insert_random_layer(neurons, acts):
    """
    Mutation by inserting random layer
    :param neurons:
    :param acts:
    :return:
    """
    new_neurons = copy.deepcopy(neurons)
    new_acts = copy.deepcopy(acts)
    ind = random.randint(0, len(acts) - 1)
    new_neurons.insert(ind, get_random_neurons())
    new_acts.insert(ind, get_activation_random())
    return new_neurons, new_acts


def random_mutation(acts, neurons):
    """
    Randomly choosing mutation
    :param acts:
    :param neurons:
    :return:
    """
    dec = random.uniform(0, 1)
    new_neurons = copy.deepcopy(neurons)
    new_acts = copy.deepcopy(acts)
    if dec < 1 / 4:
        new_neurons, new_acts = insert_random_layer(neurons, acts)
    elif 1 / 4 < dec < 1 / 2:
        new_neurons, new_acts = delete_random_layer(neurons, acts)
    elif 1 / 2 < dec < 3 / 4:
        new_acts = mutate_layer_activation(acts)
    else:
        new_neurons = mutate_layer_neurons(neurons)
    return new_neurons, new_acts
