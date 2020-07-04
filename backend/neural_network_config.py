import random


def get_random_neurons(bottom_limit=2, upper_limit=128):
    """
    Random number of neurons from bottom_limit to upper_limit
    :param bottom_limit: - lower limit
    :param upper_limit:
    :return:
    """
    return random.randint(bottom_limit, upper_limit)


def get_activations():
    """
    List of possible activations =
    opened for extension.
    :return:
    """
    return ['sigmoid', 'relu', 'tanh', 'elu', 'selu', 'softmax', 'softplus']


def get_activation_random():
    """
    Random activation for layer.
    :return:
    """
    return random.choice(get_activations())
