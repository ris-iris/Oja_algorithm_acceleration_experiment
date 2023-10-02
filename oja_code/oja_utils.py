import numpy as np

def positive_semidefined(d):
    A = np.random.randn(d,d)
    U,_,_ = np.linalg.svd(A)
    D = np.diag(1/(np.arange(d)+1))
    A = U@D@U.T
    return A

def get_stream(A):
    d = A.shape[0]
    while True:
        v = np.random.randn(d,1)
        yield A@v@v.T@A.T

def delta_lambda(A):
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = sorted(eigenvalues, reverse=True)
    return eigenvalues[0], eigenvalues[0] - eigenvalues[1]

def max_eigenvec(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_eig = np.argmax(eigenvalues)
    real_w = eigenvectors[:, max_eig].reshape(-1, 1)
    real_w /= np.linalg.norm(real_w, 2)
    return real_w