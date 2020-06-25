import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from vis.utils.utils import apply_modifications


def update_layer_activation(model, activation, index=-1):
    model.layers[index].activation = activation
    return apply_modifications(model)


def plot_(x, y, scx, scy, model):
    yhat = model.predict(x)
    x_plot = scx.inverse_transform(x)
    y_plot = scy.inverse_transform(y)
    yhat_plot = scy.inverse_transform(yhat)
    plt.scatter(x_plot, y_plot, label='Actual', s=90)
    plt.scatter(x_plot, yhat_plot, label='Predicted', edgecolors='black')
    plt.title('Input (x) versus Output (y)')
    plt.xlabel('Input Variable (x)')
    plt.ylabel('Output Variable (y)')
    plt.legend()
    plt.show()


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


def get_activation_random():
    activations_str_list = ['sigmoid', 'relu', 'tanh']  # , 'elu', 'selu', 'softmax', 'softplus']
    return random.choice(activations_str_list)


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


def get_random_model(up_lim=12, bottom_lim=1):
    layers_no = random.randint(bottom_lim, up_lim)
    acts, neurs = [], []
    for _ in range(layers_no):
        acts.append(get_activation_random())
        neurs.append(get_random_neurons())
    return create_model(neurs, acts)


def main():
    arp = argparse.ArgumentParser()
    arp.add_argument('expression', type=str)
    arp.add_argument('--linspace', type=str)
    arp.add_argument('-plot', action='store_true')
    p = arp.parse_args()
    x, y = get_data(p)
    print(x, y)
    x, y = reshape(x, y)
    scale_x, scale_y = scale_data(x, y, p.expression)
    x, y = fit_trans(x, y, scale_x, scale_y)
    print(x.min(), x.max(), y.min(), y.max())
    model = get_random_model()
    # define the loss function and optimization algorithm
    model.compile(loss='mse', optimizer='adam')
    # ft the model on the training dataset
    model.fit(x, y, epochs=500, batch_size=10, verbose=0)
    # make predictions for the input data
    yhat = model.predict(x)
    # inverse transforms
    x_plot, y_plot = inv_trans(x, y, scale_x, scale_y)
    yhat_plot = scale_y.inverse_transform(yhat)
    # report model error
    plot_model(
        model, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='LR', expand_nested=True, dpi=96
    )

    print('MSE: %.3f' % mean_squared_error(y_plot, yhat_plot))
    # plot x vs y
    pyplot.scatter(x_plot, y_plot, label='Actual')
    # plot x vs yhat
    pyplot.scatter(x_plot, yhat_plot, label='Predicted')
    pyplot.title('Input (x) versus Output (y)')
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.show()


main()
