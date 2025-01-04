import numpy as np
import detectors
from pickle import load
from matplotlib import pyplot as plt

N_test = 2**12  # Size of test dataset

snr_list = np.arange(21)  # Receiver SNR in dB

air = []
for snr in snr_list:
    file = open("detector_input/SNR_"+str(int(snr))+"_dB_k_5_m_14.pkl", "rb")
    detector_input = load(file)
    file.close()

    const = detector_input["const"]
    x = detector_input["x"]
    x = x[:, :N_test]
    x_idx = detector_input["x_idx"]
    x_idx = x_idx[:N_test]
    y = detector_input["y"]
    y = y[:, :N_test]
    var_h = detector_input["var_h"]
    mu_h = detector_input["mu_h"]
    var_n = detector_input["var_n"]

    k, const_len = const.shape
    m = x.shape[1]
    x_hat_idx = np.zeros(x.shape[1], dtype=int)
    x_hat = np.zeros((k, m))
    p_x_y_vec = []
    for i in range(m):
        p_x_y = detectors.detector_spa(const, np.atleast_2d(y[:, i]).T, var_h, mu_h, var_n)
        p_x_y_vec.append(p_x_y[0, x_idx[i]])
    air.append(1+np.mean(np.log2(p_x_y_vec))/k)


plt.plot(snr_list, air)
plt.xlabel("SNR [dB]")
plt.ylabel("AIR [bits/channel use]")
plt.show()



