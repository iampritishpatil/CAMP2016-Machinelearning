import numpy as np
import scipy.io as sio

# generating poisson data
num_neurons1=50
max_time1=10000
r1=0.1
r2=0.12
n1=(np.random.random([num_neurons1,max_time1]) <r1)
n2=(np.random.random([num_neurons1,max_time1]) <r2)
n3=(np.random.random([num_neurons1,max_time1]) <r1*2)
n3[:,max_time1/2:max_time1]=0
n4=(np.random.random([num_neurons1,max_time1]) <r2*2)
n4[:,max_time1/2:max_time1]=0
n5=np.zeros([50,10000])
r3=8.5
for i in range(50):
    sum1=0
    sum1+=int(np.rint(np.random.exponential(r3)))
    while sum1<10000:
        n5[i,sum1]=1
        sum1+=int(np.rint(np.random.exponential(np.random.exponential(r3))))
new_spiketrain=np.concatenate((n1,n2,n3,n4,n5))
new_spiketrain=new_spiketrain[np.random.permutation(new_spiketrain.shape[0]),:]
# new_spiketrain=np.concatenate((n1,n5))
sio.savemat("forclustering.mat",{"new_spiketrain":new_spiketrain})
