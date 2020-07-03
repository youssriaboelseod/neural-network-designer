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


class NeuralNetworkDesigner:
    def __init__(self, epochs=300, initial_temperature=1000, scale=0.8):
        self.epochs = epochs
        self.initial_temperature = initial_temperature
        self.scale = scale


    def draw_graph(self, x_plot, y_plot, yhat_plot, mse):
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

    def get_data(self, parsed_args):
        """
        Method convert commands line args to
        real interval with appropriate expression values
        :param parsed_args:
        :return:
        """
        lin = [int(x) for x in parsed_args.linspace[1:-1].split(',')]
        x = np.linspace(*lin)
        return x, eval(parsed_args.expression)

    def scale_data(self):
        """
        Method scale data using MinMaxScaler
        :return:
        """
        scale_x, scale_y = MinMaxScaler(), MinMaxScaler()
        return scale_x, scale_y

    def reshape(self, x, y):
        """
        Reshape vector x,y to nx1
        :param x:
        :param y:
        :return:
        """
        return x.reshape((len(x), 1)), y.reshape((len(y), 1))

    def fit_trans(self, x, y, scale_x, scale_y):
        """
        Fit scalex,scaley to x,y
        :param x:
        :param y:
        :param scale_x:
        :param scale_y:
        :return:
        """
        return scale_x.fit_transform(x), scale_y.fit_transform(y)

    def inv_trans(self, x, y, scale_x, scale_y):
        """
        Transform scale to normal vecs
        :param x:
        :param y:
        :param scale_x:
        :param scale_y:
        :return:
        """
        return scale_x.inverse_transform(x), scale_y.inverse_transform(y)

    def get_activations(self):
        """
        List of possible activations =
        opened for extension.
        :return:
        """
        return ['sigmoid', 'relu', 'tanh', 'elu', 'selu', 'softmax', 'softplus']

    def get_activation_random(self):
        """
        Random activation for layer.
        :return:
        """
        return random.choice(self.get_activations())

    def get_random_neurons(self, bottom_limit=2, upper_limit=128):
        """
        Random number of neurons from bottom_limit to upper_limit
        :param bottom_limit: - lower limit
        :param upper_limit:
        :return:
        """
        return random.randint(bottom_limit, upper_limit)

    def create_model(self, neurons_quantity_list, activations_list, krnl='he_uniform', in_dim=1, out_dim=1):
        """
        Creates model based on given lists of activations and neurons
        :param neurons_quantity_list: - neurons qunatity for each layer
        :param activations_list:  activation func for each later
        :param krnl: kernel initializer
        :param in_dim: input dismension
        :param out_dim: output dismensin
        :return:
        """
        print(neurons_quantity_list)
        print(activations_list)
        if len(neurons_quantity_list) != len(activations_list):
            raise IndexError('Here is error')
        model = Sequential()
        model.add(Dense(neurons_quantity_list[0], input_dim=in_dim, activation=activations_list[0],
                        kernel_initializer=krnl))
        for n, ac in zip(neurons_quantity_list[1:], activations_list[1:]):
            model.add(Dense(n, activation=ac, kernel_initializer=krnl))
        model.add(Dense(out_dim))
        return model

    def mutate_layer_neurons(self, neurs):
        """
        Mutation by changing quantity of neurons in random layer
        :param neurs:
        :return:
        """
        new_neurs = copy.deepcopy(neurs)
        ind = random.randint(0, len(neurs) - 1)
        new_neurs[ind] = self.get_random_neurons()
        return new_neurs

    def mutate_layer_activation(self, acts):
        """
        Mutation by changing activation function in random layer
        :param acts:
        :return:
        """
        new_acts = copy.deepcopy(acts)
        ind = random.randint(0, len(acts) - 1)
        new_acts[ind] = self.get_activation_random()
        return new_acts

    def delete_random_layer(self, neurs, acts):
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

    def insert_random_layer(self, neurs, acts):
        """
        Mutation by inserting random layer
        :param neurs:
        :param acts:
        :return:
        """
        new_neurs = copy.deepcopy(neurs)
        new_acts = copy.deepcopy(acts)
        ind = random.randint(0, len(acts) - 1)
        new_neurs.insert(ind, self.get_random_neurons())
        new_acts.insert(ind, self.get_activation_random())
        return new_neurs, new_acts

    def random_mutation(self, acts, neurs):
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
            new_neurs, new_acts = self.insert_random_layer(neurs, acts)
        elif 1 / 4 < dec < 1 / 2:
            new_neurs, new_acts = self.delete_random_layer(neurs, acts)
        elif 1 / 2 < dec < 3 / 4:
            new_acts = self.mutate_layer_activation(acts)
        else:
            new_neurs = self.mutate_layer_neurons(neurs)
        return new_neurs, new_acts

    def get_random_model_scheme(self, up_lim=25, bottom_lim=1):
        layers_no = random.randint(bottom_lim, up_lim)
        acts, neurs = [], []
        for _ in range(layers_no):
            acts.append(self.get_activation_random())
            neurs.append(self.get_random_neurons())
        return neurs, acts

    def plot_png_network(self, model):
        plot_model(
            model, to_file='model.png', show_shapes=True, show_layer_names=True,
            rankdir='LR', expand_nested=True, dpi=96
        )

    def model_prepare(self, model, x, y):
        model.compile(loss='mse', optimizer='adam')
        model.fit(x, y, epochs=300, batch_size=10, verbose=0)

    def predict_y(self, model, x, y, scale_x, scale_y):
        yhat = model.predict(x)
        x_plot, y_plot = self.inv_trans(x, y, scale_x, scale_y)
        yhat_plot = scale_y.inverse_transform(yhat)
        return yhat_plot, x_plot, y_plot

    def preprocess_data(self, args):
        x, y = self.get_data(args)
        x, y = self.reshape(x, y)
        scale_x, scale_y = self.scale_data()
        x, y = self.fit_trans(x, y, scale_x, scale_y)
        return x, y, scale_x, scale_y

    def simulated_annealing(self, t, args, save_structure=True, graph=True):
        end_time = get_time() + get_millis(t)
        T = self.initial_temperature
        scale = self.scale

        x, y, scale_x, scale_y = self.preprocess_data(args)

        neurs, acts = self.get_random_model_scheme()
        model = self.create_model(neurs, acts)
        self.model_prepare(model, x, y)
        yhat_plot, x_plot, y_plot = self.predict_y(model, x, y, scale_x, scale_y)
        mse = mean_squared_error(y_plot, yhat_plot)

        self.plot_png_network(model)
        self.draw_graph(x_plot, y_plot, yhat_plot, mse)

        print(mse)
        best_mse = mse
        best_model = model
        best_yhat = yhat_plot
        while get_time() <= end_time and T > 0:
            T *= scale
            neurs, acts = self.random_mutation(acts, neurs)
            new_model = self.create_model(neurs, acts)
            self.model_prepare(new_model, x, y)
            yhat_plot, x_plot, y_plot = self.predict_y(new_model, x, y, scale_x, scale_y)
            mse = mean_squared_error(y_plot, yhat_plot)
            if acceptance_probability(best_mse, mse, T) > random.uniform(0, 1):
                best_mse = mse
                print(best_mse)
                best_yhat = yhat_plot
                self.draw_graph(x_plot, y_plot, yhat_plot, best_mse)
                best_model = new_model
        if save_structure:
            self.plot_png_network(best_model)
        if graph:
            self.draw_graph(x_plot, y_plot, best_yhat, best_mse)
        return best_mse


def main():
    nnb = NeuralNetworkDesigner()
    arp = argparse.ArgumentParser()
    arp.add_argument('expression', type=str)
    arp.add_argument('--linspace', type=str)
    arp.add_argument('-plot', action='store_true')
    p = arp.parse_args()
    nnb.simulated_annealing(35, p)


main()
