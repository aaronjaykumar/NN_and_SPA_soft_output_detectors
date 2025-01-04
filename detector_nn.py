import numpy as np
import detectors
from pickle import load, dump
import nn
from matplotlib import pyplot as plt
import os


## --------------- Parameters for Tweaking ---------------
num_epochs = 1000               # Number of epochs for training
beta = 3                       # Learning rate
N_training = 2**12              # Size of training dataset
hidden_size1 = 10               # Size of hidden layer 1
hidden_size2 = 10               # Size of hidden layer 2
retrain_network = True          # True -> Network is freshly trained.
                                # False -> Loads weights/biases from detector_input/ if available
## -------------------------------------------------------

snr_list = np.arange(21)  # Receiver SNR in dB

p_x_y = np.zeros(np.size(snr_list))

fer = []
ser = []
air = []
for snr in snr_list:
    file = open("detector_input/SNR_"+str(int(snr))+"_dB_k_5_m_14.pkl", "rb")
    detector_input = load(file)
    file.close()

    const = detector_input["const"]
    x = detector_input["x"]
    x_idx = detector_input["x_idx"]
    x_label = detector_input["x_label"]
    y = detector_input["y"]
    var_h = detector_input["var_h"]
    mu_h = detector_input["mu_h"]
    var_n = detector_input["var_n"]

    k, const_len = const.shape
    m = x.shape[1]
    x_hat_idx = np.zeros(x.shape[1], dtype=int)
    x_hat = np.zeros((k, m))

    model = detectors.DetectorNN(k+3, hidden_size1, hidden_size2, 2**k)
    criterion = nn.CELossSoftmax()

    # Training Phase
    y_train = np.concatenate((y[:, :N_training], var_n*np.ones((1, N_training)), var_h*np.ones((1, N_training)),
                             mu_h*np.ones((1, N_training))), axis=0)
    x_train = x[:, :N_training]
    x_label_train = x_label[:, :N_training]

    if retrain_network or (not os.path.isfile("trained_networks/snr_" + str(int(snr)) + "dB.pkl")):
        for epoch in range(num_epochs):
            outputs = model(y_train)
            loss = criterion(x_label_train, outputs, model)
            model.zero_grad()
            model.backward()
            model.step(beta)
        network_weights = {"fc1_w": model.fc1.w,
                           "fc1_b": model.fc1.b,
                           "fc2_w": model.fc2.w,
                           "fc2_b": model.fc2.b,
                           "fc3_w": model.fc3.w,
                           "fc3_b": model.fc3.b,
                           }
        file = open("trained_networks/snr_" + str(int(snr)) + "dB.pkl",
                    "wb")
        dump(network_weights, file)
        file.close()

    else:
        file = open("trained_networks/snr_" + str(int(snr)) + "dB.pkl",
                    "rb")
        network_weights = load(file)
        file.close()
        model.fc1.w = network_weights["fc1_w"]
        model.fc1.b = network_weights["fc1_b"]
        model.fc2.w = network_weights["fc2_w"]
        model.fc2.b = network_weights["fc2_b"]
        model.fc3.w = network_weights["fc3_w"]
        model.fc3.b = network_weights["fc3_b"]

    # Test Phase
    x_test = x[:, N_training:]
    N_test = x_test.shape[1]
    y_test = np.concatenate((y[:, N_training:], var_n*np.ones((1, N_test)), var_h*np.ones((1, N_test)),
                             mu_h*np.ones((1, N_test))), axis=0)
    x_label_test = x_label[:, N_training:]
    x_idx_test = x_idx[N_training:]

    outputs = model(y_test)

    # Soft-Decision Detection
    air.append(1+np.mean(np.sum(x_label_test*np.log2(outputs), axis=0))/k)

plt.plot(snr_list,air)
plt.xlabel("SNR [dB]")
plt.ylabel("AIR [bits/channel use]")
plt.show()




