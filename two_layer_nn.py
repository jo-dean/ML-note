
def layer_sizes(X, Y):
    n_x = X.shape[0] # size of input layer
    n_h = 4 # size  of hidden layer
    n_y = Y.shape[0] # size of output layer_sizes
    return (n_x, n_h, n_y)
    
def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"w1": w1,
                  "b1":b1,
                  "w2":w2,
                  "b2":b2}
    return parameters
    
    
def forward_propagation(X, parameters):
    # 获取参数
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    # 前向传播
    Z1 = np.dot(w1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2,A1) + b2
    A2 = sigmoid(Z2)
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache

# 损失函数
def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of example
    #计算交叉熵损失
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -1/m * np.sum(logprobs)
    # 确保代价函数的维度
    cost = np.squeeze(cost)

#反向传播
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    #计算dw1,db1,dw2,db2
    dZ2 = A2 - Y
    dw2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dw1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dz1, axis = 1, keepdims = True)
    
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    return grads
    
def update_parameters(parameters, grads, learning_rate = 1.2):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    dw1 = grads["dw1"]
    db1 = grads["db1"]
    dw2 = grads["dw2"]
    db2 = grads["db2"]
    
    w1 -= dw1 * learning_rate
    b1 -= db1 * learning_rate
    w2 -= dw2 * learning_rate
    b2 -= db2 * learning_rate
    
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    return parameters
    
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    # 梯度下降
    for i in range(0, num_iterations):
        # 前向计算 input:"X,parameters", outputs:"A2, cache"
        A2,cache = forward_propagation(X, parameters)
        # 代价函数 input:"A2, Y" output: "cost"
        cost = compute_cost(A2, Y, parameters)
        # 反向传播 input:"parameters, cache, X,Y" outputs:"grads"
        grads = backward_propagation(parameters, cache, X, Y)
        # 梯度下降参数更新 input:"parameters, grads" outputs: "parameters"
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        # 每1000次迭代打印一次损失
        if print_cost and i %1000 == 0:
            print( "Cost after iteration %i: %f" %(i, cost))
            
    return parameters

    
    
    
    
    
    
    
    
    
    
    
    
    
    
