import theano
import theano.tensor as T
import numpy as np

def generate_random_clouds(n_points, n_classes, n_dim):
    
    N = 1000
    K = 1
    feats = 2
    classes = 2
    
    points = np.zeros(((N * K * classes), feats + 1))
    for c in range(K * classes):
        mean = np.random.random(2)*5
        
        # create a Covariance matrix
        cov = (0.7 * np.eye(n_dim, n_dim) + \
            0.3 * np.random.random((n_dim, n_dim))) * 0.4
        cov = np.dot(cov, cov.T)  # make it positive semidefinite
        
        points[c * N:(c + 1) * N, :-1] = np.random.multivariate_normal(mean, cov, N)
        points[c * N:(c + 1) * N, -1] = c % classes
    
    permutation = np.random.permutation(N * K * classes)
    X, Y = points[permutation, :-1].astype(np.float32), \
           points[permutation, -1].astype(np.int32)
         
    return X, Y  


