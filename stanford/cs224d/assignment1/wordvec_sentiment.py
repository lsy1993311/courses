__author__ = 'guoxy'
import random
import numpy as np

__all__ = ['softmax', 'sigmoid', 'sigmoid_grad',
           'forward_backward_prop']


def softmax(x):
    """ Softmax function """

    # Since x may have only one dimension while we need max and sum operation
    # along axis 1, to avoid checking the dimension number, we convert them
    # to operation along axis 0 by transposing the array
    e = np.exp(x.T - np.max(x.T, axis=0)).T
    prob = (e.T / np.sum(e.T, axis=0)).T

    return prob


def sigmoid(x):
    """ Sigmoid function """

    y = np.clip(x, a_min=-15, a_max=15)
    y = 1 / (1 + np.exp(-y))

    return y


def sigmoid_grad(f):
    """ Sigmoid gradient function """

    g = f * (1 - f)

    return g


def forward_backward_prop(data, labels, params, dimensions):
    """ Forward and backward propagation for a two-layer sigmoidal network """
    ###################################################################
    # Compute the forward propagation and for the cross entropy cost, #
    # and backward propagation for the gradients for all parameters.  #
    ###################################################################

    ### Unpack network parameters (do not modify)
    t = 0
    W1 = np.reshape(params[t:t+dimensions[0]*dimensions[1]], (dimensions[0], dimensions[1]))
    t += dimensions[0]*dimensions[1]
    b1 = np.reshape(params[t:t+dimensions[1]], (1, dimensions[1]))
    t += dimensions[1]
    W2 = np.reshape(params[t:t+dimensions[1]*dimensions[2]], (dimensions[1], dimensions[2]))
    t += dimensions[1]*dimensions[2]
    b2 = np.reshape(params[t:t+dimensions[2]], (1, dimensions[2]))

    ### YOUR CODE HERE: forward propagation

    z1 = data.dot(W1) + b1
    h = sigmoid(z1)
    z2 = h.dot(W2) + b2
    yhat = softmax(z2)
    cost = -np.log((yhat * labels)[labels == 1]).sum()

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    gradz2 = yhat
    gradz2[labels == 1] -= 1
    gradz1 = W2.dot(gradz2.T).T * sigmoid_grad(h)

    gradW1 = data.T.dot(gradz1)
    if len(gradz1.shape) > 1:
        gradb1 = gradz1.sum(axis=0)
    else:
        gradb1 = gradz1
    gradW2 = h.T.dot(gradz2)
    if len(gradz2.shape) > 1:
        gradb2 = gradz2.sum(axis=0)
    else:
        gradb2 = gradz2

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def softmaxCostAndGradient(predicted, target, outputVectors):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector (\hat{r} in #
    #           the written component)                                #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       #
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    prob = softmax(predicted.T.dot(outputVectors))
    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE

    ### END YOUR CODE

    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of no more than 2*C strings, the context #
    #             words                                               #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE

    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
    ###################################################################
    # Implement the continuous bag-of-words model in this function.   #
    # Input/Output specifications: same as the skip-gram model        #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE

    ### END YOUR CODE

    return cost, gradIn, gradOut