import pickle
import numpy as np

def gaussian_kernel(x, mu, variance): 
    return np.exp(-((x - mu)**2 / (2*variance)))

def make_design_matrix(x, means, variance=0.25): 
    """
    Purpose: make N x M matrix: 
    - N is # of data samples
    - M is number of features given by Gaussian kernels
    """

    X = np.zeros((len(x), len(means)))

    # apply Gaussian kernels to create design matrix
    for i, mean in enumerate(means): 
        X[:, i] = gaussian_kernel(x, mean, variance)

    return X

def main(data): 

    x = data[:, 0]
    y = data[:, 1]
    means = np.arange(0, 7, 1)
    means = 1.25 * means - 3.75

    # create design matrix
    X = make_design_matrix(x, means)

    H = X @ X.T

    print(H.shape)

    egv, _ = np.linalg.eig(H)
    lambda_max = float(np.max(egv))
    eps = 2./lambda_max
    print(f"{eps=}")


    

if __name__ == "__main__": 
    
    # (1000, 2) where n=1000 for (x, y) coordinates
    data = np.load('hw1_p1.npy')
    main(data)
    
