import theano
import theano.tensor as T
import numpy as np

def generate_random_clouds(n_points, n_classes, n_dim, n_mixtures=1):

    points = np.zeros(((n_points * n_mixtures * n_classes), n_dim + 1))
    for c in range(n_mixtures * n_classes):
        mean = np.random.random(2)*5
        
        # create a Covariance matrix
        cov = (0.7 * np.eye(n_dim, n_dim) + \
            0.3 * np.random.random((n_dim, n_dim))) * 0.4
        cov = np.dot(cov, cov.T)  # make it positive semidefinite
        
        points[c * n_points:(c + 1) * n_points, :-1] = np.random.multivariate_normal(mean, cov, n_points)
        points[c * n_points:(c + 1) * n_points, -1] = c % n_classes
    
    permutation = np.random.permutation(n_points * n_mixtures * n_classes)
    X, Y = points[permutation, :-1].astype(np.float32), \
           points[permutation, -1].astype(np.int32)
         
    return X, Y  


