import random
from backend.neural_network_config import *
def mutate_layer_neurons(neurs):
    """
    Mutation by changing quantity of neurons in random layer
    :param neurs:
    :return:
    """
    new_neurs = copy.deepcopy(neurs)
    ind = random.randint(0, len(neurs) - 1)
    new_neurs[ind] = get_random_neurons()
    return new_neurs


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


def delete_random_layer(neurs, acts):
    """
    Mutation by deleting random layer
    :param neurs:
    :param acts:
    :return:
    """
    new_neurs = copy.deepcopy(neurs)
    new_acts = copy.deepcopy(acts)
    ind = random.randint(0, len(acts) - 1)
    del new_neurs[ind]
    del new_acts[ind]
    return new_neurs, new_acts


def insert_random_layer(neurs, acts):
    """
    Mutation by inserting random layer
    :param neurs:
    :param acts:
    :return:
    """
    new_neurs = copy.deepcopy(neurs)
    new_acts = copy.deepcopy(acts)
    ind = random.randint(0, len(acts) - 1)
    new_neurs.insert(ind, get_random_neurons())
    new_acts.insert(ind, get_activation_random())
    return new_neurs, new_acts


def random_mutation(acts, neurs):
    """
    Randomly choosing mutation
    :param acts:
    :param neurs:
    :return:
    """
    dec = random.uniform(0, 1)
    new_neurs = copy.deepcopy(neurs)
    new_acts = copy.deepcopy(acts)
    if dec < 1 / 4:
        new_neurs, new_acts = insert_random_layer(neurs, acts)
    elif 1 / 4 < dec < 1 / 2:
        new_neurs, new_acts = delete_random_layer(neurs, acts)
    elif 1 / 2 < dec < 3 / 4:
        new_acts = mutate_layer_activation(acts)
    else:
        new_neurs = mutate_layer_neurons(neurs)
    return new_neurs, new_acts
