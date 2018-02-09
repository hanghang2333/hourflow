import hourflow as hf
from hourflow.graph import *
from hourflow.optimizer import *
import hourflow.Operation.operations as ops
Graph().as_default()
'''
A = Variable([[1,0],[0,-1]])
b = Variable([1,1])

x = placeholder()
y = ops.matmul(A,x)
z = ops.add(y,b)
z = ops.reduce_sum(z)

session = Session()
output = session.run(z,{x:[1,2]})
print(output)
'''

X = placeholder()
c = placeholder()
W = Variable(np.random.randn(2,2))
b = Variable(np.random.randn(2))
p = ops.softmax(ops.add(ops.matmul(X,W),b))
J = ops.negative(ops.reduce_sum(ops.reduce_sum(ops.multiply(c,ops.log(p)),axis=1)))
minimization_op = GradientDescentOptimizer(learning_rate=0.01).minimize(J)
red_points = np.random.randn(50,2)-2*np.ones((50,2))
blue_points = np.random.randn(50,2)+2*np.ones((50,2))
feed_dict = {X:np.concatenate((blue_points,red_points)),
             c:[[1,0]]*len(blue_points)+[[0,1]]*len(red_points)}
sess = Session()
for step in range(1000):
    J_value = sess.run(J,feed_dict)
    if step%10 == 0:
        print("Step:",step,"Loss:",J_value.shape)
    sess.run(minimization_op,feed_dict)
W_value = sess.run(W)
print("Weight matrix:\n",W_value)
b_value = sess.run(b)
print("Bias:\n",b_value)
#'''