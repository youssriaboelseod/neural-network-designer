import copy
import random
import time

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from backend.ModelEditor import *
from backend.Tooling import *


def get_millis(t):
    return t * 1000


def get_time():
    return int(round(get_millis(time.time())))


def acceptance_probability(cost, new_cost, temp):
    """
    Decide if we move to new soultion.
    :param cost:
    :param new_cost:
    :param temp:
    :return:
    """
    return 1 if new_cost < cost else np.exp(- (new_cost - cost) / temp)


def get_random_model_scheme(up_lim=25, bottom_lim=1):
    layers_no = random.randint(bottom_lim, up_lim)
    acts, neurs = [], []
    for _ in range(layers_no):
        acts.append(get_activation_random())
        neurs.append(get_random_neurons())
    return neurs, acts



def create_model(neurons_quantity_list, activations_list, krnl='he_uniform', in_dim=1, out_dim=1):
    """
    Creates model based on given lists of activations and neurons
    :param neurons_quantity_list: - neurons qunatity for each layer
    :param activations_list:  activation func for each later
    :param krnl: kernel initializer
    :param in_dim: input dismension
    :param out_dim: output dismensin
    :return:
    """
    if len(neurons_quantity_list) != len(activations_list):
        raise IndexError('Here is error')
    model = Sequential()
    model.add(Dense(neurons_quantity_list[0], input_dim=in_dim, activation=activations_list[0],
                    kernel_initializer=krnl))
    for n, ac in zip(neurons_quantity_list[1:], activations_list[1:]):
        model.add(Dense(n, activation=ac, kernel_initializer=krnl))
    model.add(Dense(out_dim))
    return model


def preprocess_data(args):
    x, y = get_data(args)
    x, y = reshape(x, y)
    scale_x, scale_y = scale_data()
    x, y = fit_trans(x, y, scale_x, scale_y)
    return x, y, scale_x, scale_y


def predict_y(model, x, y, scale_x, scale_y):
    yhat = model.predict(x)
    x_plot, y_plot = inv_trans(x, y, scale_x, scale_y)
    yhat_plot = scale_y.inverse_transform(yhat)
    return yhat_plot, x_plot, y_plot


class NeuralNetworkDesigner:
    def __init__(self, all_neurons, all_activations, linspace, expression, epochs=300, initial_temperature=1000,
                 scale=0.8):
        self.epochs = epochs
        self.initial_temperature = initial_temperature
        self.scale = scale
        self.layers_neurons = all_neurons
        self.layers_activations = all_activations
        self.x, self.y = get_data_direct(linspace, expression)
        self.x, self.y = reshape(self.x, self.y)
        self.scale_x, self.scale_y = scale_data()
        self.x, self.y = fit_trans(self.x, self.y, self.scale_x, self.scale_y)

    def model_prepare(self, model, x, y):
        model.compile(loss='mse', optimizer='adam')
        model.fit(x, y, epochs=self.epochs, batch_size=10, verbose=0)

    def simulated_annealing(self, t, args, save_structure=True, graph=True, resets=True):
        end_time = get_time() + get_millis(t)
        T = self.initial_temperature
        scale = self.scale

        neurs, acts = copy.deepcopy(self.layers_neurons), copy.deepcopy(self.layers_activations)
        model = create_model(neurs, acts)
        self.model_prepare(model, self.x, self.y)
        yhat_plot, x_plot, y_plot = predict_y(model, self.x, self.y, self.scale_x, self.scale_y)
        mse = mean_squared_error(y_plot, yhat_plot)

        plot_png_network(model)
        draw_graph(x_plot, y_plot, yhat_plot, mse)

        best_mse = mse
        best_model = model
        best_yhat = yhat_plot
        while get_time() <= end_time and T > 0:
            T *= scale
            neurs, acts = random_mutation(acts, neurs)
            new_model = create_model(neurs, acts)
            self.model_prepare(new_model, self.x, self.y)
            yhat_plot, x_plot, y_plot = predict_y(new_model, self.x, self.y, self.scale_x, self.scale_y)
            mse = mean_squared_error(y_plot, yhat_plot)
            if acceptance_probability(best_mse, mse, T) > random.uniform(0, 1):
                best_mse = mse
                print(best_mse)
                best_yhat = yhat_plot
                draw_graph(x_plot, y_plot, yhat_plot, best_mse)
                best_model = new_model
        if save_structure:
            plot_png_network(best_model)
        if graph:
            draw_graph(x_plot, y_plot, best_yhat, best_mse)
        return best_mse
