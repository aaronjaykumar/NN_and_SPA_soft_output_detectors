import numpy as np
import nn


def normpdf(x, mu, var):
    return 1/np.sqrt(2*np.pi*var)*np.exp(-1/2*(x-mu)**2/var)


def detector_spa(const, y, var_h, mu_h, var_n):

    m = mu_h
    var = var_h
    pxy_all = []
    for const_vec in const.T:
        pxy = 1
        for i in range(const_vec.shape[0]):
            pxy = pxy * 0.5 * normpdf(const_vec[i]*y[i], m, var+var_n)
            m = (const_vec[i]*y[i]*var + mu_h*var_n)/(var+var_n)
            var = ((var * var_n)/(var + var_n)) + var_h
        pxy_all.append(pxy)
    p_y = sum(pxy_all)
    p_x_given_y_all = [pxy / p_y for pxy in pxy_all]
    p_x_given_y_all = np.array(p_x_given_y_all)
    p_x_given_y_all = np.transpose(p_x_given_y_all)
    return p_x_given_y_all


class DetectorNN(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DetectorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x, self)
        x = self.sigmoid1(x, self)
        x = self.fc2(x, self)
        x = self.sigmoid2(x, self)
        x = self.fc3(x, self)
        x = self.softmax(x, self)
        return x

