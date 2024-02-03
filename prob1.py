import pickle
import numpy as np
import matplotlib.pyplot as plt

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

    # solve for upper bound of learning rate epsilon (problem 1.3)
    H = X @ X.T
    egv, _ = np.linalg.eig(H)
    lambda_max = float(np.max(egv))
    eps = 2./lambda_max
    print(f"{eps=}")

    losses = []
    gradient_norm = []
    
    # train loop for 1k epochs
    w = np.zeros(len(means))
    for epoch in range(1000): 
        y_hat = X @ w
        loss = 0.5 * np.mean((y - y_hat)**2)
        losses.append(loss)
        grad = X.T @ (y_hat - y)
        current_grad_norm = np.linalg.norm(grad)**2
        gradient_norm.append(current_grad_norm)     
        w -= eps * grad

    # Plot loss
    plt.figure()  # Create a new figure
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over 1000 epochs')
    plt.savefig('prob1_loss.png')

    # plot convergence
    # FIXME: 
    log_grad = np.log(gradient_norm)
    t_values = np.arange(1, 1001)
    convergence_upper_bound = np.log(2 * lambda_max * (log_grad[0] - log_grad[-1]) / t_values)
    plt.figure()
    plt.plot(log_grad, label='Log of Norm-Squared of Gradient')
    plt.plot(convergence_upper_bound, label='Log of Convergence Upper Bound', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Log of Norm-Squared of Gradient vs. Convergence Upper Bound')
    plt.savefig('prob1_convergence.png')

    # Plot predictions against data, sort predictions
    plt.figure()  # Create a new figure for the predictions plot
    y_hat = X @ w
    idx = np.argsort(x)
    plt.plot(x[idx], y_hat[idx], label='Predictions', color='red')
    plt.scatter(x, y, label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Predictions vs. Data')
    plt.savefig('prob1_predictions.png')

if __name__ == "__main__": 
    
    # (1000, 2) where n=1000 for (x, y) coordinates
    data = np.load('hw1_p1.npy')
    main(data)
    
