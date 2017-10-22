import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    # print("i:{0}, y[i]:{1}".format(i, y[i]))
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i,:]
        dW[:,j] += X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  
  loss /= num_train
  dW   /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW   += reg * 2 * W 
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]
  num_class=W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  S=X.dot(W)
  Sy=S[range(num_train),y]
  # print(type(S), type(Sy)) #, S type:{1} , type(S)
  # print("Sy:{0}, S:{1}".format(Sy.shape, S.shape))
  L = S-Sy[:,np.newaxis]+1
  # L = Sy-S
  # L = L+1
  # print(L.shape)
  L[range(num_train), y]=0
  loss = np.sum(np.maximum(L,0))/num_train
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # A=np.zeros( (num_train, num_class) )
  # oneMtx=np.ones( (num_train, num_class))
  # A[range(num_train), y]=1
  # Lb = (L>0)
  # C = np.concatenate([ np.identity(num_class), np.zeros( (num_train-num_class, num_class)) ], axis=0)
  # # My= np.repeat(np.bincount(y)[:,np.newaxis], num_class, axis=1)
  # My= np.repeat(Sy[:,np.newaxis], num_class, axis=1)
  # # print(C)
  # # print(C.shape)
  # op1=np.dot(np.transpose(X),C)
  # op2=np.dot(np.transpose(Lb),C)
  # dW = np.dot(op1, op2)

  # dWy= np.dot(np.transpose(X), Lb)
  # dW-= dWy
  # mid=np.dot(op1,op2) #D*c
  # dW =  - np.dot(mid, np.transpose(My))


  Lb = (L>0)
  # print(Lb.shape)
  dS =np.zeros((num_train, num_class))
  dS[Lb]=1
  dSy=np.zeros((num_train, num_class))
  dSy[range(num_train), y]=np.sum(Lb,axis=1)
  dS-=dSy
  dW =X.T.dot(dS)
  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
