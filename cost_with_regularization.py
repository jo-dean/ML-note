import dnn

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    w3 = parameters["w3"]
    
    # cross_entropy part of the cost
    cross_entropy_cost = compute_cost(A3, Y)
    L2_regularization_cost = 1/m * lambd/2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)))
    cost = cross_entropy_cost + L2_regularization_cost
    return cost
    
def backward_propagation_with_regularization(X, Y, cache, lambd)
    m = X.shape[1]
    (Z1, A1, w1, b1, Z2, w2, b2, Z3, A3, w3, b3) = cache
    
    dZ3 = A3 - Y
    dw3 = 1./m * np.dot(dZ3, A2.T) +  lambd/m * w3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)

    dA2 = np.dot(w3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))

    dw2 = 1./m * np.dot(dZ2, A1.T) + lambd/m * w2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(w2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))

    dw1 = 1./m * np.dot(dZ1, X.T) + lambd/m * w1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dw3": dw3, "db3": db3,"dA2": dA2,"dZ2": dZ2, "dw2": dw2, "db2": db2, "dA1": dA1, "dZ1": dZ1, "dw1": dw1, "db1": db1}    
    return gradients
