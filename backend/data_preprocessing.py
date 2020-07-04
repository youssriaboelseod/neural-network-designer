from sklearn.preprocessing import MinMaxScaler


def scale_data():
    """
    Method scale data using MinMaxScaler
    :return:
    """
    scale_x, scale_y = MinMaxScaler(), MinMaxScaler()
    return scale_x, scale_y


def reshape(x, y):
    """
    Reshape vector x,y to nx1
    :param x:
    :param y:
    :return:
    """
    return x.reshape((len(x), 1)), y.reshape((len(y), 1))


def fit_trans(x, y, scale_x, scale_y):
    """
    Fit scalex,scaley to x,y
    :param x:
    :param y:
    :param scale_x:
    :param scale_y:
    :return:
    """
    return scale_x.fit_transform(x), scale_y.fit_transform(y)


def inv_trans(x, y, scale_x, scale_y):
    """
    Transform scale to normal vecs
    :param x:
    :param y:
    :param scale_x:
    :param scale_y:
    :return:
    """
    return scale_x.inverse_transform(x), scale_y.inverse_transform(y)
