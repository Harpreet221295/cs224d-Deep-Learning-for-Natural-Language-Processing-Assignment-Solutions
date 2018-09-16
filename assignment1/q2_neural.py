import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))


    layer1_output = np.dot(data,W1)+b1
    layer1_activations = sigmoid(layer1_output)

    output_scores = np.dot(layer1_activations,W2)+b2
    softmax_scores = softmax(output_scores)

    cross_entropy_loss = -1 * np.sum(labels*np.log(softmax_scores))

    #print cross_entropy_loss.shape

    cost = cross_entropy_loss
    

    doutput_scores = softmax_scores
    
    #doutput_scores-=labels

    label_index = np.argmax(labels, axis=1)

    doutput_scores[np.arange(data.shape[0]), label_index] -= 1

    gradW2 = np.dot(layer1_activations.T, doutput_scores)
    gradb2 = np.sum(doutput_scores,axis=0)

    dlayer1_activations = np.dot(doutput_scores,W2.T)

    dlayer1_output = sigmoid_grad(layer1_activations)*dlayer1_activations

    gradW1 = np.dot(data.T, dlayer1_output)
    gradb1 = np.sum(dlayer1_output,axis=0)
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()