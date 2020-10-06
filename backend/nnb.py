from sklearn.metrics import mean_squared_error
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from backend.data_preprocessing import *
from backend.model_editing import *
from backend.neural_network_config import *
from backend.tools import *


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
    """

    :param up_lim: Build random scheme
    :param bottom_lim:
    :return:
    """
    layers_no = random.randint(bottom_lim, up_lim)
    acts, neurons = [], []
    for _ in range(layers_no):
        acts.append(get_activation_random())
        neurons.append(get_random_neurons())
    return neurons, acts


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


def nantonum(yhat_plot, y_plot):
    for i in range(len(y_plot)):
        for j in range(len(y_plot[i])):
            y_plot[i][j] = np.nan_to_num(y_plot[i][j])
    for i in range(len(yhat_plot)):
        for j in range(len(yhat_plot[i])):
            yhat_plot[i][j] = np.nan_to_num(yhat_plot[i][j])


class NeuralNetworkDesigner:
    def __init__(self, all_neurons, all_activations, lin_space, expression, epochs=300, initial_temperature=1000,
                 scale=0.8, resets=True):
        self.data = dict()
        self.data['epochs'] = epochs
        self.data['initial_activations'] = all_activations
        self.data['initial_neurons'] = all_neurons
        self.data['linspace'] = lin_space
        self.data['expression'] = expression
        self.data['initial_temperature'] = initial_temperature
        self.data['scale'] = scale
        self.data['resets'] = resets
        self.x, self.y = get_data_direct(lin_space, expression)
        self.x, self.y = reshape(self.x, self.y)
        self.scale_x, self.scale_y = scale_data()
        self.x, self.y = fit_trans(self.x, self.y, self.scale_x, self.scale_y)

    def model_prepare(self, model, x, y):
        model.compile(loss='mse', optimizer='adam')
        model.fit(x, y, epochs=self.data['epochs'], batch_size=10, verbose=0)

    def predict_y(self, model):
        yhat = model.predict(self.x)
        x_plot, y_plot = inv_trans(self.x, self.y, self.scale_x, self.scale_y)
        yhat_plot = self.scale_y.inverse_transform(yhat)
        return yhat_plot, x_plot, y_plot

    def simulated_annealing(self, t, save_structure=True, graph=True, step_limit=300000):
        self.data['time'] = t
        wh = get_millis(int(t))
        end_time = get_time() + wh
        temp = self.data['initial_temperature']
        scale = self.data['scale']

        neurs, acts = copy.deepcopy(self.data['initial_neurons']), copy.deepcopy(self.data['initial_activations'])
        model = create_model(neurs, acts)
        self.model_prepare(model, self.x, self.y)
        yhat_plot, x_plot, y_plot = self.predict_y(model)
        nantonum(yhat_plot, y_plot)
        mse = np.nan_to_num(mean_squared_error(np.nan_to_num(y_plot), np.nan_to_num(yhat_plot)))
        plot_png_network(model)
        draw_graph(x_plot, y_plot, yhat_plot, mse)
        self.data['initial_yhat'] = yhat_plot
        best_mse = mse
        self.data['first_mse'] = copy.deepcopy(mse)
        print(best_mse)
        best_model = model
        best_neurs, best_acts = neurs, acts
        best_yhat = yhat_plot
        step = 0
        while get_time() <= end_time and temp > 0:
            temp *= scale

            new_neurs, new_acts = get_random_model_scheme() if self.data[
                                                                   'resets'] and step % step_limit else random_mutation(
                acts, neurs)
            new_model = create_model(new_neurs, new_acts)
            self.model_prepare(new_model, self.x, self.y)
            yhat_plot, x_plot, y_plot = self.predict_y(new_model)
            nantonum(yhat_plot, y_plot)
            new_mse = mean_squared_error(y_plot, yhat_plot)

            if acceptance_probability(best_mse, mse, temp) > random.uniform(0, 1):
                neurs, acts = new_neurs, new_acts
                model = new_model
                mse = new_mse

                if mse < best_mse:
                    best_mse = mse
                    best_neurs, best_acts = neurs, acts
                    print(best_mse)
                    best_yhat = yhat_plot
                    draw_graph(x_plot, y_plot, yhat_plot, best_mse)
                    best_model = model
            step += 1
            if step % step_limit and self.data['first_mse'] == best_mse:
                print('rs')
                neurs, acts = get_random_model_scheme()

        if save_structure:
            plot_png_network(best_model)
        if graph:
            draw_graph(x_plot, y_plot, best_yhat, best_mse)
        self.data['x_plot'] = x_plot
        self.data['y_plot'] = y_plot
        self.data['yhat_plot'] = best_yhat
        self.data['best_mse'] = best_mse
        self.data['best_neurons'] = best_neurs
        self.data['best_activations'] = best_acts
        return best_mse, best_neurs, best_acts
