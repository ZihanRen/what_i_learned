#%%
# KL ( p||Q )  =  \sum { P(x)log(P(x))  /  Q(x) }

events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]

# calculate the kl divergence
from math import log2

def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))


a = sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
print(a)

# %%
