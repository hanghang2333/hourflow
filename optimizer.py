from queue import Queue
from .Operation.operations import *
class GradientDescentOptimizer:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate
    
    def minimize(self,loss):
        learning_rate = self.learning_rate
        class MinimizationOperation(Operation):
            def compute(self):
                grad_table = compute_gradients(loss)
                for node in grad_table:
                    if type(node)==Variable:
                        grad = grad_table[node]
                        node.value -= learning_rate*grad
        return MinimizationOperation()

_gradient_registry = {}
class RegisterGradient:
    def __init__(self,op_type):
        self._op_type = eval(op_type)
    def __call__(self,f):
        _gradient_registry[self._op_type] = f
        return f

def compute_gradients(loss):
    grad_table = {}
    grad_table[loss] = 1
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)
    while not queue.empty():
        node = queue.get()
        if node!=loss:
            grad_table[node] = 0
            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]
                lossgrads_wrt_consumer_inputs = bprop(consumer,lossgrad_wrt_consumer_output)
                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_consumer_inputs
                else:
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    grad_table[node] += lossgrad_wrt_node
        if hasattr(node,"input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)
    return grad_table

@RegisterGradient("negative")
def _negative_gradient(op,grad):
    return -grad
@RegisterGradient("log")
def _log_gradient(op,grad):
    x = op.inputs[0]
    return grad/(x)
@RegisterGradient("multiply")
def _multiply_gradient(op,grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad*B,grad*A]
@RegisterGradient("matmul")
def _matmul_gradient(op,grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad.dot(B.T),A.T.dot(grad)]
@RegisterGradient("sigmoid")
def _sigmoid_gradient(op,grad):
    sigmoid = op.inputs[0]
    return sigmoid*(1-sigmoid)*grad

@RegisterGradient("add")
def _add_gradient(op,grad):
    a = op.inputs[0]
    b = op.inputs[1]
    grad_wrt_a = grad
    while np.ndim(grad_wrt_a)>len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a,axis=0)
    for axis,size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a,axis=axis,keepdims=True)
    grad_wrt_b = grad
    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b,axis=0)
    for axis,size in enumerate(b.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b,axis=axis,keepdims=True)
    return [grad_wrt_a,grad_wrt_b]
@RegisterGradient("reduce_sum")
def _reduce_sum_gradient(op,grad):
    A = op.inputs[0]
    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape//output_shape
    grad = np.reshape(grad,output_shape)
    return np.tile(grad,tile_scaling)
@RegisterGradient("softmax")
def _softmax_gradient(op,grad):
    softmax = op.output
    return (grad-np.reshape(np.sum(grad*softmax,-1),[-1,1]))*softmax

@RegisterGradient("flatten")
def _flatten_gradient(op,grad):
    return np.reshape(grad,op.inputs[0].shape)

@RegisterGradient("conv2d")
def _conv2d_gradient(op, grad):
    x,w,b = op.inputs[0],op.inputs[1],op.inputs[2]
    dout = grad
    #print(grad)
    N, F, H1, W1 = dout.shape
    N, C, H, W = x.shape
    HH = w.shape[2]
    WW = w.shape[3]
    S = op.strides
    pad = op.pad
    PH,PW = 0,0
    Ho,Wo = int(1+(H-HH)/S),int(1+(W-WW)/S)
    x_pad = None
    dx_pad = None
    if pad == 'valid':
        x_pad = np.zeros((N,C,H,W))
        dx_pad = np.zeros((N,C,H,W))
        x_pad[:,:,0:H,0:W]=x
    else:
        PH = (H-1)*S+HH-H
        PW = (W-1)*S+WW-W#当求出来需要pad的数目是奇数时就不好处理，这里选择右边多pad一个
        x_pad = np.zeros((N,C,H+PH,W+PW))
        x_pad[:,:,int(PH/2):int(PH/2)+H,int(PW/2):int(PW/2)+W]=x                                               
        Ho,Wo = H,W
    dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
    dx_pad = np.zeros((N,C,H+PH,W+PW))
    db = np.sum(dout, axis=(0,2,3))
    for n in range(N):
        for i in range(H1):
            for j in range(W1):
            # Window we want to apply the respective f th filter over (C, HH, WW)
                x_window = x_pad[n, :, i * S : i * S + HH, j * S : j * S + WW]
                for f in range(F):
                    dw[f] += x_window * dout[n, f, i, j]
                    dx_pad[n, :, i * S : i * S + HH, j * S : j * S + WW] += w[f] * dout[n, f, i, j]
    dx = dx_pad[:, :, PH:PH+H, PW:PW+W]
    return [dx, dw, db]

@RegisterGradient("pooling")
def _pooling_gradient(op,grad):
    size = op.size
    stride = op.stride
    mode = op.mode
    x = op.inputs[0]
    batch,cha,in_row,in_col = np.shape(x)
    # outputMap sizes
    out_row,out_col = int(np.floor(in_row/stride)),int(np.floor(in_col/stride))
    row_remainder,col_remainder = np.mod(in_row,stride),np.mod(in_col,stride)
    if row_remainder != 0:
        out_row +=1
    if col_remainder != 0:
        out_col +=1
    # padding
    temp_map = np.lib.pad(x, ((0,0),(0,0),(0,size-row_remainder),(0,size-col_remainder)),'edge')
    # max pooling
    out = np.zeros_like(x)
    for batchi in range(batch):
        for chai in range(cha):
            for r_idx in range(0,out_row):
                for c_idx in range(0,out_col):
                    startX = c_idx * stride
                    startY = r_idx * stride
                    poolField = temp_map[batchi,chai,startY:startY + size, startX:startX + size]
                    poolOut = np.max(poolField)
                    index = np.where(poolField==poolOut)
                    #print(index)
                    try:
                        indexx = index[0][0]+startY
                        indexy = index[1][0]+startX
                    except Exception:
                        indexx = startY
                        indexy = startX
                    #print('ss',grad.shape,batchi,chai,out_row,out_col)
                    out[batchi,chai,indexx,indexy] = grad[batchi][chai][r_idx][c_idx]
    return  out