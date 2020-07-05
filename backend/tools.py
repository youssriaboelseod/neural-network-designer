import time

import numpy as np
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model


def get_millis(t):
    return t * 1000


def get_time():
    return int(round(get_millis(time.time())))


def plot_png_network(model):
    plot_model(
        model, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='LR', expand_nested=True, dpi=96
    )


def draw_graph(x_plot, y_plot, yhat_plot, mse):
    """
    Method responsible for drawing plot.
    :param x_plot:
    :param y_plot:
    :param yhat_plot:
    :param mse:
    :return:
    """
    pyplot.scatter(x_plot, y_plot, label='Actual')
    pyplot.scatter(x_plot, yhat_plot, label='Predicted')
    pyplot.title('MSE: %.3f' % mse)
    pyplot.xlabel('Input Variable (x) ')
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


def get_data_direct(space, expr):
    """
    Method convert directly commands line args to
    real interval with appropriate expression values
    :param expr:
    :param space:
    :return:
    """
    lin = [int(x) for x in space[1:-1].split(',')]
    x = np.linspace(*lin)
    return x, eval(expr)
