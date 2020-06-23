import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


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


def get_random_model(x, y, ep=500):
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=ep, batch_size=10, verbose=0)
    return model


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
    # learn to use mean sometimes we have nan so we changing it to means.
    return scale_x.fit_transform(x), scale_y.fit_transform(y)


def inv_trans(x, y, scale_x, scale_y):
    return scale_x.inverse_transform(x), scale_y.inverse_transform(y)


def main():
    arp = argparse.ArgumentParser()
    arp.add_argument('expression', type=str)
    arp.add_argument('--linspace', type=str)
    arp.add_argument('-plot', action='store_true')
    p = arp.parse_args()
    x, y = get_data(p)
    x, y = reshape(x, y)
    scale_x, scale_y = scale_data(x, y, p.expression)
    x, y = fit_trans(x, y, scale_x, scale_y)
    print(x.min(), x.max(), y.min(), y.max())
    # design the neural network model
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1))
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
