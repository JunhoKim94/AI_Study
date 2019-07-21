import numpy as np

"""
DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!! 

[Description]
__init__ - Initialize necessary variables for optimizer class
input   : gamma, epsilon
return  : X

update   - Update weight for one minibatch
input   : w - current weight, grad - gradient for w, lr - learning rate
return  : updated weight 
"""

class SGD:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        pass


        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        updated_weight = w - lr * grad


        # =============================================================
        return updated_weight

class Momentum:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        self.epsilon = epsilon
        self.v = 1
        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        self.v = self.gamma * self.v + lr * grad
        updated_weight = w - self.v



        # =============================================================
        return updated_weight


class RMSProp:
    # ========================= EDIT HERE =========================
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        self.epsilon = epsilon
        self.G = 1

        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        self.G = self.gamma*self.G + (1-self.gamma)*np.square(grad)
        #print(self.G.shape,grad.shape)
        updated_weight = w - (lr/np.sqrt(self.epsilon+self.G))*grad



        # =============================================================
        return updated_weight