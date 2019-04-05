# coding:utf-8
import numpy as np


def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    # 网络层数
    L = len(layer_dims)

    
    for l in range(1, L):
        print(l)
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        print(parameters["w" + str(l)].shape)
        print(layer_dims[l], layer_dims[l-1])
        print(parameters["b" + str(l)].shape)
        print(layer_dims[l], 1)
        
    
    #print(parameters["w" + str(l)].shape)
    #print(layer_dims[l], layer_dims[l-1])
    assert(parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
    #print(parameters["b" + str(l)].shape)
    #print(layer_dims[l], 1)
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

# example
parameters = initialize_parameters_deep([5,4,3])
print("w1 = " + str(parameters["w1"]))
print("b1 = " + str(parameters["b1"]))
print("w2 = " + str(parameters["w2"]))
print("b2 = " + str(parameters["b2"]))


def linear_activation_forward(A_prev, w, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, w, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, w, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (w.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    cache = []
    A = X
    # 网络层数
    L = len(parameters)
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["w" + str(l)], parameters["b" + str(l)], "relu")
        cache.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["w"+str(l)], parameters["b" + str(l)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y,np.log(1-AL)))/m
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

def linear_backward(dZ, cache):
    A_prev, w, b = cachesm
    m = A_prev.shape[1]
    dw = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(w.T, dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    
    return dA_prev, dw, db
    
def linear_activation_backward(dA, cache,activation):
    linear_cache, activation_cache = caches
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dw, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dw, db = linear_backward(dZ, linear_cache)
    return dA_prev, dw, db
    
def L_model_forward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L), grads["dw" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")]
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dw_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dw" + str(l + 1)] = dw_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
    

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)
    
    for l in range(L):
        parameters["w" + str(l+1)] = parameters["w" + str(l+1)] - learning_rate*grads["dw" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters

def L_layer_model(X,Y,layer_dims,learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    np.random.seed(1)
    costs = []
    
    parameters = initialize_parameters_deep(layer_dims)
    
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y,caches)
        
        parameters = update_parameters(parameters,grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after ineration %i: %f" %(i, cost)) 
            costs.append(cost)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per tens)")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    
    return parameters
    
    

























