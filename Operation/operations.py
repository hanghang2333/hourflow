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
        #print('softmax',a_value)
        return np.exp(a_value)/(np.sum(np.exp(a_value),axis=1)[:,None])

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

class flatten(Operation):
    def __init__(self,x):
        super().__init__([x])
    def compute(self,x_value):
        #print('d',x_value)
        return np.reshape(x_value,[x_value.shape[0],-1]) 

class conv2d(Operation):
    def __init__(self,x,w,b, strides=1, pad='same'):
        #    H' = 1 + (H + 2 * pad - HH) / stride
        #    W' = 1 + (W + 2 * pad - WW) / stride  
        super().__init__([x,w,b])
        self.strides = strides
        self.pad = pad
    def compute(self,x,w,b):
        N,C,H,W = x.shape#10*224*224*3
        F,_,HH,WW = w.shape#6*3*3*3
        S = self.strides
        PH,PW = 0,0
        Ho,Wo = int(1+(H-HH)/S),int(1+(W-WW)/S)
        x_pad = None
        if self.pad == 'valid':
            x_pad = np.zeros((N,C,H,W))
            x_pad[:,:,0:H,0:W]=x
        else:
            PH = (H-1)*S+HH-H
            PW = (W-1)*S+WW-W#当求出来需要pad的数目是奇数时就不好处理，这里选择右边多pad一个
            if PH%2==0 and PW%2==0:
                x_pad = np.zeros((N,C,H+PH,W+PW))
                #print(PH/2,PH/2+H,PW/2,PW/2+W)
                x_pad[:,:,int(PH/2):int(PH/2)+H,int(PW/2):int(PW/2)+W]=x                
            elif PH%2==0 and PW%2==1:
                x_pad = np.zeros((N,C,H+PH,W+PW))
                x_pad[:,:,int(PH/2):int(PH/2)+H,int(PW/2):int(PW/2)+W]=x
            elif PH%2==1 and PW%2==0:
                x_pad = np.zeros((N,C,H+PH,W+PW))
                x_pad[:,:,int(PH/2):int(PH/2)+H,int(PW/2):int(PW/2)+W]=x
            else:
                x_pad = np.zeros((N,C,H+PH,W+PW))
                x_pad[:,:,int(PH/2):int(PH/2)+H,int(PW/2):int(PW/2)+W]=x                                
            Ho,Wo = H,W
        #x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')
        out = np.zeros((N,F,Ho,Wo))
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                    out[:,f,i,j] = np.sum(x_pad[:, :, i*S : i*S+HH, j*S : j*S+WW] * w[f, :, :, :], axis=(1, 2, 3)) 
            out[:,f,:,:]+=b[f]
        return out

class pooling(Operation):
    def __init__(self,x,size=2,stride=2,mode='max'):
        #目前只可以使用maxpooling
        super().__init__([x])
        self.size = size
        self.stride = stride
        self.mode = mode
    def compute(self,x):
        batch,cha,in_row,in_col = np.shape(x)
        # outputMap sizes
        out_row,out_col = int(np.floor(in_row/self.stride)),int(np.floor(in_col/self.stride))
        row_remainder,col_remainder = np.mod(in_row,self.stride),np.mod(in_col,self.stride)
        if row_remainder != 0:
            out_row +=1
        if col_remainder != 0:
            out_col +=1
        outputMap = np.zeros((batch,cha,out_row,out_col))
        # padding
        temp_map = np.lib.pad(x, ((0,0),(0,0),(0,self.size-row_remainder),(0,self.size-col_remainder)),'edge')
        # max pooling
        for batchi in range(batch):
            for chai in range(cha):
                for r_idx in range(0,out_row):
                    for c_idx in range(0,out_col):
                        startX = c_idx * self.stride
                        startY = r_idx * self.stride
                        poolField = temp_map[batchi,chai,startY:startY + self.size, startX:startX + self.size]
                        poolOut = np.max(poolField)
                        outputMap[batchi,chai,r_idx,c_idx] = poolOut
        return  outputMap