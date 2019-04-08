def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5)
    np.random.seed(1)  #检索参数
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    w3 = parameters["w3"]
    b3 = parameters["b3"]
    
    # linear -> relu ->linear -> relu -> linear -> sigmoid
    Z1 = np.dot(w1, X) + b1
    A1 = relu(Z1)
    
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob
    A1 = np.multiply(D1, A1)
    A1 = A1 / keep_prob
    
    Z2 = np.dot(w2, A1) + b2
    A2 = relu(Z2)
    
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = np.multiply(D2, A2)
    A2 = A2 / keep_prob
    
    Z3 = np.dot(w3 ,A2) + b3
    A3 = relu(Z3)
    
    cache = (Z1, D1, A1, w1, b1, Z2, D2, A2, w2, b2, Z3, A3, w3 , b3)
    return A3, cache
    
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, w1, b1, Z2, D2, A2, w2, b2, Z3, A3, w3, b3) = cache
    
    dZ3 = A3 - Y
    dw3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis = 1, keepdims = True)
    dA2 = np.dot(w3.T, dZ3)
    
    dA2 = np.multiply(dA2, D2)
    dA2 = dA2 / keep_prob
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dw2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(w2.T, dZ2)

    dA1 = np.multiply(dA1, D1)   
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,"dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}    
    return gradients
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
