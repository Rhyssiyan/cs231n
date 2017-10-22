import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N  = X.shape[0]
  C  = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # forward pass
  Z=X.dot(W) #N*C
  ZMax=np.max(Z,axis=1)
  # Z[range(N),y]-=0
  Z-=ZMax[:,np.newaxis]
  eZ=np.exp(Z)  # N*C
  eZSum=np.sum(eZ,axis=1)
  yp=np.zeros((N,C))
  dZ=np.zeros((N,C))
  for i in range(N):
    for j in range(C):    
      yp[i]=eZ[i]/eZSum[i]
      if j==y[i]:
        dZ[i,j]=1-yp[i,j]
      else:
        dZ[i,j]=-yp[i,j]
  dW=-X.T.dot(dZ)/N
  dW+=reg*W
  # eZSoftmaxDeno=np.sum(eZ,axis=1)[:,np.newaxis]
  # yp=1-eZ/eZSoftmaxDeno #N*C
  # yp[range(N),y] = -yp[range(N),y]+1
  loss=np.sum(-np.log(yp[range(N),y]))
  loss/=N
  loss+=0.5*reg*np.sum(W*W)

  # y_pred=np.argmax(yp, axis=1) #N*1
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N  = X.shape[0]
  C  = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!     +dZ[range(N),y]                                                      #
  #############################################################################
  #calculate loss
  Z=X.dot(W)
  ZMax=np.max(Z,axis=1)
  Z-=ZMax[:,np.newaxis]
  eZ=np.exp(Z)  # N*C
  eZSum=np.sum(eZ,axis=1)
  yp=eZ/eZSum[:,np.newaxis]
  # minimum=np.min(yp)
  # if minimum<1e-6:
  #   print(minimum)
  #   print("type yp: ", type(yp[0][0]))
  #   print("type eZ: ", type(eZ[0][0]))  
  #   print("type eZSum: ", type(eZSum[0]))
  mid=-np.log(yp[range(N), y])
  loss=np.sum(mid)/N
  # print(N)
  loss+=0.5*reg*np.sum(W*W)

  #calculate gradient
  dZ=-yp
  dZ[range(N),y]+=1
  dW=-X.T.dot(dZ)/N
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

