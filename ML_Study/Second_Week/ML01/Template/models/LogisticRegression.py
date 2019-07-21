import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================
        num = x.shape[0]
        #x = x[:,1:]
        #print(x.shape,y.shape)
        #print(epochs,num%batch_size)
        for epoch in range(epochs):
            for steps in range(num // batch_size):
                W_T = np.transpose(self.W)
                #print(self.W.shape,W_T.shape)
                #self.W.reshape(1,self.num_features)
                X = x[batch_size*steps:batch_size*(steps+1),:]
                Y = y[batch_size*steps:batch_size*(steps+1)]
                #X = X.reshape(len(X[0,:]),len(X[:,0]))
                X = np.transpose(X)
                #print(X.shape)
                hypothesis = self._sigmoid(Y - np.matmul(W_T,X))
                loss = Y*np.log(hypothesis)+(1-Y)*np.log(1-hypothesis)
                cost = -(1/batch_size)*np.sum(loss)
                #print(steps)
                grad =  np.sum(hypothesis*(1-hypothesis)*X,axis=1)
                #grad = grad.reshape(8,1)
                #print(grad.shape,self.W.shape)
                self.W = optim.update(self.W, grad, lr)
            if epoch % 1000 == 0:
                print("weight:",self.W)
                print("Loss:",cost)
        final_loss = cost




        # ============================================================
        return final_loss

    def eval(self, x):
        threshold = 0.5
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        W_T = np.transpose(self.W)
        x_T = np.transpose(x)
        p = self._sigmoid(np.matmul(W_T,x_T))
        if p > threshold:
            pred = 1
        else:
            pred = 0



        # ============================================================

        return pred

    def _sigmoid(self, x):
        sigmoid = 1/(1+np.exp(-x))

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================




        # ============================================================
        return sigmoid