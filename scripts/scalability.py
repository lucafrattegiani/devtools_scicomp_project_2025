from pyclassify.utils import distance_numpy, distance_numba
import time
import numpy as np
import matplotlib.pyplot as plt

#Time storage:
tnb = []
tnp = []

#Tested dimensionalities:
dl = 10 ** np.arange(1, 9)

#Number of runs:
runs = 10

#Loop over dimensionalities:
for d in dl:
    nb_mean = []
    np_mean = []

    #Multiple trials for each dimensionality:
    for run in range(runs):

        #Generate random vectors:
        x1 = np.random.random_sample((d, ))
        x2 = np.random.random_sample((d, ))

        #Numba distance:
        t0_nb = time.time()
        dnb = distance_numba(x1, x2)
        t1_nb = time.time()

        #Numpy distance:
        t0_np = time.time()
        dnp = distance_numpy(x1, x2)
        t1_np = time.time()

        #Track distances:
        nb_mean.append(t1_nb - t0_nb)
        np_mean.append(t1_np - t0_np)

    #Take the mean:
    nb_mean = np.array(nb_mean).mean()
    np_mean = np.array(np_mean).mean()

    #Append:
    tnb.append(nb_mean)
    tnp.append(np_mean)

fig = plt.figure()
plt.title('Scalability plot')
plt.xlabel('d')
plt.ylabel('t (s)')
plt.scatter(dl, tnp, c = "blue", label = 'numpy')
plt.plot(dl, tnp, c = "blue")
plt.scatter(dl, tnb, c = "red", label = 'numba')
plt.plot(dl, tnb, c = "red")
plt.legend()
fig.savefig('./logs/scalability.png')
plt.xscale('log')
fig.savefig('./logs/scalability_log.png')