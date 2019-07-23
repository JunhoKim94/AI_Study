import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features -1 
        self.W = np.random.random((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        #final_loss = None   # loss of final epoch
        
        # Training should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================
        num = x.shape[0]
        x = x[:,1:]
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

                loss = np.square(Y - np.matmul(X,self.W))/2
                loss_sum = (1/batch_size)*np.sum(loss)
                #print(steps)
                grad =  -np.sum(np.dot((Y-np.matmul(X,self.W)).T,X),axis=1)
                #grad = grad.reshape(8,1)
                #print(grad,self.W.shape)
                self.W = optim.update(self.W, grad, lr)
            if epoch % 1000 == 0:
                print("weight:",self.W)
                print("Loss:",loss_sum)
        final_loss = loss_sum


        # ============================================================
        return final_loss

    def eval(self, x):
        x = x[:,1:]

        pred = np.matmul(x,self.W).reshape(x.shape[0])

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================



        # ============================================================
        return pred
