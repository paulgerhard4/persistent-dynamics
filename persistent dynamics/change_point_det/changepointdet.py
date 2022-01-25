import matplotlib.pyplot as plt
import numpy as np
import gudhi
from gtda.time_series import takens_embedding_optimal_parameters
from functions import *
import openml
from openml.datasets.functions import get_dataset

#Rössler x componente
L = []
with open('rossler.txt') as txtfile:
    for line in txtfile:
        L.append(float(line.rstrip()))

Rossler = L[-10000:]

#Lorenz System
point_cloud = get_dataset(42182).get_data(dataset_format='array')[0]
X = point_cloud[:, 2]
Lorenz = X

lis = []
for x in Lorenz:
	lis.append(x)
for x in Rossler:
	lis.append(x)
for x in Lorenz:
	lis.append(x)
plt.plot(lis)
plt.show()

n = 2500
final = [lis[i * n:(i + 1) * n] for i in range((len(lis) + n - 1) // n )] 

zs = []
ys = []

def max_emb_dim(ts):
	maxi = 2
	for interval in ts:
		d = takens_embedding_optimal_parameters(interval, 100, 5)
		if d[1] > maxi:
			maxi = d[1]

	return maxi

m = max_emb_dim(final)

for x in range(0,len(final)-1):
	for h in [0,1]:
		
		hom1 = persistence(final[x], emdim = m, tilay = 20, homdim = [h])
		hom1g = []
		for a in hom1:
			for b in a:
				hom1g.append([b[0],b[1]])
		hom2 = persistence(final[x+1], emdim = m, tilay = 20, homdim = [h])
		hom2g = []
		for a in hom2:
			for b in a:
				hom2g.append([b[0],b[1]])

		db1 = gudhi.bottleneck_distance(hom1g,hom2g)
		if h == 0:
			zs.append(db1)
		if h == 1:
			ys.append(db1)

plt.plot(zs)
plt.plot(ys)
plt.legend(['H0','H1'])
plt.show()
