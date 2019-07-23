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
            rand = np.arange(x.shape[0])
            np.random.shuffle(rand)
            #shuffle the data
            x = x[rand]
            y = y[rand]

            for steps in range(num // batch_size):

                X = x[batch_size*steps:batch_size*(steps+1),:]
                Y = y[batch_size*steps:batch_size*(steps+1)].reshape(batch_size,1)

                hypothesis = self._sigmoid(np.matmul(X,self.W))
                
                loss = Y*np.log(hypothesis)+(1-Y)*np.log(1-hypothesis)
                cost = -(1/batch_size)*np.sum(loss)

                grad =  np.sum(np.dot((-Y+hypothesis).T,X),axis=1)/batch_size

                self.W = optim.update(self.W, grad, lr)
            if epoch % 1000 == 0:
                print(grad)
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

        p = self._sigmoid(np.matmul(x,self.W))
        pred = 1*(p > threshold).reshape(x.shape[0])


        # ============================================================

        return pred

    def _sigmoid(self, x):
        sigmoid = 1/(1+np.exp(-x))

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================




        # ============================================================
        return sigmoid