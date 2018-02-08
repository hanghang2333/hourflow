from ..graph import *
class add(Operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    
    def compute(self,x_value,y_value):
        return x_value+y_value


class matmul(Operation):
    def __init__(self,a,b):
        super().__init__([a,b])
    
    def compute(self,a_value,b_value):
        return a_value.dot(b_value)

class sigmoid(Operation):
    def __init__(self,a):
        super().__init__([a])
    
    def compute(self,a_value):
        return 1/(1+np.exp(-a_value))

class softmax(Operation):
    def __init__(self,a):
        super().__init__([a])
    
    def compute(self,a_value):
        return np.exp(a_value)/np.sum(np.exp(a_value),axis=1)[:,None]

class log(Operation):
    def __init__(self,a):
        super().__init__([a])
    
    def compute(self,a_value):
        return np.log(a_value)

class multiply(Operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    
    def compute(self,x_value,y_value):
        return x_value*y_value

class reduce_sum(Operation):
    def __init__(self,A,axis=None):
        super().__init__([A])
        self.axis=axis
    
    def compute(self,A_value):
        return np.sum(A_value,self.axis)

class negative(Operation):
    def __init__(self,x):
        super().__init__([x])
    
    def compute(self,x_value):
        return -x_value
