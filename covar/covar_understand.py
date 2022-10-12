#%%
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[128,128],porosity=0.3)

f = plt.figure()
plt.imshow(im,cmap='gray')
plt.axis('off')
plt.show()

def visualize_matrix(matrix):
    f = plt.figure()
    plt.imshow(matrix)
    plt.axis('off')
    plt.show()


# covariance matrix of image
covar = np.cov(im,rowvar=False)
visualize_matrix(covar)


# %% LU decomposition
import scipy
P,L,U = scipy.linalg.lu(covar)
visualize_matrix(P)
visualize_matrix(L)
visualize_matrix(U)

# %% simulation
norm = np.random.normal(0,1,size=(128,1000))
sample_img = L.dot(norm)
plt.imshow(sample_img,cmap='gray')
# %%
import numpy as np

no_obs = 1000             # Number of observations per column
means = [1, 2, 3]         # Mean values of each column
no_cols = 3               # Number of columns

sds = [1, 2, 3]           # SD of each column
sd = np.diag(sds)         # SD in a diagonal matrix for later operations

observations = np.random.normal(0, 1, (no_cols, no_obs)) # Rd draws N(0,1) in [3 x 1,000]

cor_matrix = np.array([[1.0, 0.6, 0.9],
                       [0.6, 1.0, 0.5],
                       [0.9, 0.5, 1.0]])          # The correlation matrix [3 x 3]

cov_matrix = np.dot(sd, np.dot(cor_matrix, sd))   # The covariance matrix

Chol = np.linalg.cholesky(cov_matrix)             # Cholesky decomposition


sam_eq_mean = Chol .dot(observations)             # Generating random MVN (0, cov_matrix)

s = sam_eq_mean.transpose() + means               # Adding the means column wise
samples = s.transpose()                           # Transposing back