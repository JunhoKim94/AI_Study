import numpy as np

class SoftmaxClassifier:
    def __init__(self, num_features, num_label):
        self.num_features = num_features
        self.num_label = num_label
        self.W = np.random.random((self.num_features, self.num_label))

    def train(self, x, y, epochs, batch_size, lr, optimizer):
        """
        N : # of training data
        D : # of features
        C : # of classes

        [INPUT]
        x : (N, D), input data (first column is bias for all data)
        y : (N, )
        epochs: (int) # of training epoch to execute
        batch_size : (int) # of minibatch size
        lr : (float), learning rate
        optimizer : (Python class) Optimizer

        [OUTPUT]
        final_loss : (float) loss of last training epoch

        [Functionality]
        Given training data, hyper-parameters and optimizer, execute training procedure.
        Training should be done in minibatch (not the whole data at a time)
        Procedure for one epoch is as follow:
        - For each minibatch
            - Compute probability of each class for data
            - Compute softmax loss
            - Compute gradient of weight
            - Update weight using optimizer
        * loss of one epoch = Mean of minibatch losses
        (minibatch losses = [0.5, 1.0, 1.0, 0.5] --> epoch loss = 0.75)

        """
        print('========== TRAINING START ==========')
        final_loss = None   # loss of final epoch
        num_data, num_feat = x.shape
        print(num_feat)
        losses = []
        for epoch in range(1, epochs + 1):
            batch_losses = []   # list for storing minibatch losses
            rand = np.arange(num_data)
            np.random.shuffle(rand)
            x = x[rand]
            y = y[rand]
        # ========================= EDIT HERE ========================
            for steps in range(num_data//batch_size):
                X = x[batch_size*steps:batch_size*(steps+1),:]
                Y = y[batch_size*steps:batch_size*(steps+1)]

                hypothesis = self._softmax(np.matmul(X,self.W)) #가로축 방향으로 softmax 진행
                #need to one-hot encoding
                encod = np.eye(3)
                one_hot = encod[Y[:]]
                #loss fun
                loss = self.softmax_loss(hypothesis,one_hot)/batch_size
                batch_losses.append(loss)
                grad = self.compute_grad(X,self.W,hypothesis,one_hot)/batch_size
                self.W =optimizer.update(self.W,grad,lr)
    


        # ============================================================
            epoch_loss = sum(batch_losses) / len(batch_losses)  # epoch loss
            # print loss every 10 epoch
            if epoch % 1000 == 0:
                print('Epoch %d : Loss = %.4f' % (epoch, epoch_loss))
                print(self.W)
            # store losses
            losses.append(epoch_loss)
        final_loss = losses[-1]

        return final_loss

    def eval(self, x):
        """

        [INPUT]
        x : (N, D), input data

        [OUTPUT]
        pred : (N, ), predicted label for N test data

        [Functionality]
        Given N test data, compute probability and make predictions for each data.
        """
        pred = None
        # ========================= EDIT HERE ========================
        hypothesis = self._softmax(np.matmul(x,self.W))
        hypothesis[:, np.argmax(hypothesis ,axis = 1)]





        # ============================================================
        return pred

    def softmax_loss(self, prob, label):
        """
        N : # of minibatch data
        C : # of classes

        [INPUT]
        prob : (N, C), probability distribution over classes for N data
        label : (N, ), label for each data

        [OUTPUT]
        softmax_loss : scalar, softmax loss for N input

        [Functionality]
        Given probability and correct label, compute softmax loss for N minibatch data
        """
        softmax_loss = 0.0
        # ========================= EDIT HERE ========================

        loss = -np.sum(label * np.log(prob))
        softmax_loss = loss





        # ============================================================
        return softmax_loss

    def compute_grad(self, x, weight, prob, label):
        """
        N : # of minibatch data
        D : # of features
        C : # of classes

        [INPUT]
        x : (N, D), input data
        weight : (D, C), Weight matrix of classifier
        prob : (N, C), probability distribution over classes for N data
        label : (N, ), label for each data. (0 <= c < C for c in label)

        [OUTPUT]
        gradient of weight: (D, C), Gradient of weight to be applied (dL/dW)

        [Functionality]
        Given input (x), weight, probability and label, compute gradient of weight.
        """
        grad_weight = np.zeros_like(weight, dtype=np.float32) # (D, C)
        # ========================= EDIT HERE ========================

        grad = prob[:, np.argmax(prob ,axis = 1)] - 1
        
        grad = np.eye(grad.shape[0]) * grad
        grad = np.sum(grad,axis = 0)
        grad = x*grad.reshape(len(grad),1)
        grad_weight = np.sum(grad , axis = 0).reshape(weight.shape[0],1)


        # ============================================================
        return grad_weight


    def _softmax(self, x):
        """
        [INPUT]
        x : (N, C), score before softmax

        [OUTPUT]
        softmax : (same shape with x), softmax distribution over axis-1

        [Functionality]
        Given an input x, apply softmax function over axis-1 (classes).
        """
        softmax = None
        # ========================= EDIT HERE ========================
        x1 = np.exp(x)
        x2 = np.sum(np.exp(x), axis = 1)
        x2 = x2.reshape(len(x2),1)
        softmax = x1/x2




        # ============================================================
        return softmax