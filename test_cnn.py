import hourflow as hf
from hourflow.graph import *
from hourflow.optimizer import *
import hourflow.Operation.operations as ops

def makeonehot(y):
    yout = np.zeros([y.shape[0],10])
    for idx,i in enumerate(y):
        yout[idx][int(i)] = 1
    return yout
#make data
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
images,target = mnist.data,mnist.target
images = np.reshape(images,[-1,28,28])
images = images[:,np.newaxis,:,:]
target = np.reshape(target,[-1,1])
from sklearn.utils import shuffle
images,target= shuffle(images,target,random_state=0)
images = images/256
target = target
#images = np.reshape(images,[-1,784])
target = makeonehot(target)
#print(target)
#print(images[0])
num = len(images)
cl = int(num*0.9)
trainx,trainy,testx,testy = images[0:cl],target[0:cl],images[cl:],target[cl:]
print('trainsize:',len(trainx),'testsize:',len(testx))

Graph().as_default()
X = placeholder()
y = placeholder()

conv1_W = Variable(np.random.randn(4,1,3,3)*1e-2)
conv1_b = Variable(np.random.randn(4)*1e-2)
conv1 = ops.sigmoid(ops.conv2d(X,conv1_W,conv1_b))#28*28
pool1 = ops.pooling(conv1)#14*14

conv2_W = Variable(np.random.randn(8,4,3,3)*1e-2)
conv2_b = Variable(np.random.randn(8)*1e-2)
conv2 = ops.sigmoid(ops.conv2d(pool1,conv2_W,conv2_b))#14*14
pool2 = ops.pooling(conv2)#7*7

conv3_W = Variable(np.random.randn(8,8,3,3)*1e-2)
conv3_b = Variable(np.random.randn(8)*1e-2)
conv3 = ops.sigmoid(ops.conv2d(pool2,conv3_W,conv3_b))#7*7
pool3 = ops.pooling(conv3)#4*4

flatten = ops.flatten(pool3)
dense1_w = Variable(np.random.randn(128,10)*1e-2)
dense1_b = Variable(np.random.randn(10)*1e-2)
dense = ops.sigmoid(ops.add(ops.matmul(flatten,dense1_w),dense1_b))
'''
flatten = ops.flatten(X)
dense1_w = Variable(np.random.randn(784,100)*1e-3)
dense1_b = Variable(np.random.randn(100)*1e-3)
dense1 = ops.sigmoid(ops.add(ops.matmul(flatten,dense1_w),dense1_b))
dense2_w = Variable(np.random.randn(100,10)*1e-3)
dense2_b = Variable(np.random.randn(10)*1e-3)
dense = ops.add(ops.matmul(dense1,dense2_w),dense2_b)
'''
p = ops.softmax(dense)
J = ops.negative(ops.reduce_sum(ops.reduce_sum(ops.multiply(y,ops.log(p)),axis=1)))
minimization_op = GradientDescentOptimizer(learning_rate=0.001).minimize(J)

feed_dict_train = {X:trainx,y:trainy}
feed_dict_test = {X:testx,y:testy}
sess = Session()
batch_size = 32
for step in range(1000):
    J_value = 0
    for i in range(int(len(trainx)/batch_size)):
        nowx = trainx[i*batch_size:(i+1)*batch_size]
        nowy = trainy[i*batch_size:(i+1)*batch_size]
        feed_dict_train = {X:nowx,y:nowy}
        J_value = sess.run(J,feed_dict_train)
        sess.run(minimization_op,feed_dict_train)
        if i%1==0:
            print("step:",step,"batch",i,'trainloss',J_value)
    #if step%1 == 0:
    #    print("Step:",step,"TrainLoss:",J_value)
    #if step%100==0:
    #    pass
        #J_test = sess.run(J,feed_dict_test)
        #print('Step:',step,"TestLoss:",J_test)