import argparse
import copy
import random
import time

import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from vis.utils.utils import apply_modifications


def acceptance_probability(cost, new_cost, temp):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temp)
        return p


def update_layer_activation(model, activation, index=-1):
    model.layers[index].activation = activation
    return apply_modifications(model)


def draw_graph(x_plot, y_plot, yhat_plot):
    pyplot.scatter(x_plot, y_plot, label='Actual')
    pyplot.scatter(x_plot, yhat_plot, label='Predicted')
    pyplot.title('Input (x) versus Output (y)')
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.show()


def get_data(parsed_args):
    """
    Method convert commands line args to
    real interval with appropriate expression values
    :param parsed_args:
    :return:
    """
    lin = [int(x) for x in parsed_args.linspace[1:-1].split(',')]
    x = np.linspace(*lin)
    return x, eval(parsed_args.expression)


def scale_data(x, y, expr):
    # if any(['sin', 'cos']) in expr:
    #    print(expr)
    # learn to use mean sometimes we have nan so we changing it to means.
    scale_x, scale_y = MinMaxScaler(), MinMaxScaler()
    return scale_x, scale_y


def reshape(x, y):
    return x.reshape((len(x), 1)), y.reshape((len(y), 1))


def fit_trans(x, y, scale_x, scale_y):
    return scale_x.fit_transform(x), scale_y.fit_transform(y)


def inv_trans(x, y, scale_x, scale_y):
    return scale_x.inverse_transform(x), scale_y.inverse_transform(y)


def get_activations():
    return ['sigmoid', 'relu', 'tanh']  # , 'elu', 'selu', 'softmax', 'softplus']


def get_activation_random():
    return random.choice(get_activations())


def get_random_neurons(bottom_limit=2, upper_limit=128):
    return random.randint(bottom_limit, upper_limit)


def create_model(neurons_quantity_list, activations_list, krnl='he_uniform', in_dim=1, out_dim=1):
    if len(neurons_quantity_list) != len(activations_list):
        raise IndexError('')
    model = Sequential()
    model.add(Dense(neurons_quantity_list[0], input_dim=in_dim, activation=activations_list[0],
                    kernel_initializer=krnl))
    for n, ac in zip(neurons_quantity_list[1:], activations_list[1:]):
        model.add(Dense(n, activation=ac, kernel_initializer=krnl))
    model.add(Dense(out_dim))
    return model


def mutate_layer_neurons(neurs):
    new_neurs = copy.deepcopy(neurs)
    ind = random.randint(0, len(neurs) - 1)
    new_neurs[ind] = get_random_neurons()
    return new_neurs


def mutate_layer_activation(acts):
    new_acts = copy.deepcopy(acts)
    ind = random.randint(0, len(acts) - 1)
    new_acts[ind] = get_activation_random()
    return new_acts


def delete_random_layer(neurs, acts):
    new_neurs = copy.deepcopy(neurs)
    new_acts = copy.deepcopy(acts)
    ind = random.randint(0, len(acts) - 1)
    del new_neurs[ind]
    del new_acts[ind]
    return new_neurs, new_acts


def insert_random_layer(neurs, acts):
    new_neurs = copy.deepcopy(neurs)
    new_acts = copy.deepcopy(acts)
    ind = random.randint(0, len(acts) - 1)
    new_neurs.insert(ind, get_random_neurons())
    new_acts.insert(ind, get_activation_random())
    return new_neurs, new_acts


def random_mutation(acts, neurs):
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


def get_random_model_scheme(up_lim=12, bottom_lim=1):
    layers_no = random.randint(bottom_lim, up_lim)
    acts, neurs = [], []
    for _ in range(layers_no):
        acts.append(get_activation_random())
        neurs.append(get_random_neurons())
    return neurs, acts


def plot_png_network(model):
    plot_model(
        model, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='LR', expand_nested=True, dpi=96
    )


def get_millis(t):
    return t * 1000


def get_time():
    return int(round(get_millis(time.time())))


def model_prepare(model, x, y):
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=300, batch_size=10, verbose=0)


def predict_y(model, x, y, scale_x, scale_y):
    yhat = model.predict(x)
    x_plot, y_plot = inv_trans(x, y, scale_x, scale_y)
    yhat_plot = scale_y.inverse_transform(yhat)
    return yhat_plot, x_plot, y_plot


def simulated_annealing(t, args, T0=1000, scale=0.8, save_structure=True, graph=True):
    end_time = get_time() + get_millis(t)
    T = T0

    x, y = get_data(args)
    x, y = reshape(x, y)
    scale_x, scale_y = scale_data(x, y, args.expression)
    x, y = fit_trans(x, y, scale_x, scale_y)

    neurs, acts = get_random_model_scheme()
    model = create_model(neurs, acts)
    model_prepare(model, x, y)
    yhat_plot, x_plot, y_plot = predict_y(model, x, y, scale_x, scale_y)
    mse = mean_squared_error(y_plot, yhat_plot)

    plot_png_network(model)
    draw_graph(x_plot, y_plot, yhat_plot)

    print(mse)
    best_mse = mse
    best_pair = (neurs, acts)
    best_model = model
    while get_time() <= end_time and T > 0:
        T *= scale
        new_neurs, new_acts = random_mutation(acts, neurs)
        new_model = create_model(new_neurs, new_acts)
        model_prepare(new_model, x, y)
        yhat_plot, x_plot, y_plot = predict_y(new_model, x, y, scale_x, scale_y)
        mse = mean_squared_error(y_plot, yhat_plot)
        if acceptance_probability(best_mse, mse, T) > random.uniform(0, 1):
            best_mse = mse
            print(best_mse)
            draw_graph(x_plot, y_plot, yhat_plot)
            best_pair = (new_neurs, new_acts)
            best_model = new_model
    if save_structure:
        plot_png_network(model)
    if graph:
        draw_graph(x_plot, y_plot, yhat_plot)
    return best_mse


def main():
    arp = argparse.ArgumentParser()
    arp.add_argument('expression', type=str)
    arp.add_argument('--linspace', type=str)
    arp.add_argument('-plot', action='store_true')
    p = arp.parse_args()
    simulated_annealing(15, p)


main()
