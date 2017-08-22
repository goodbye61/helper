

import scipy.misc as sm
from cnn import *
import numpy as np
from solver import *
import matplotlib.pyplot as plt 
from bn_cnn import *
import time 
from new import *
from kshnet import *
from bb_kshnet import *

#####################################################################################
################### THIS IS MAIN CODE FOR HELPER ####################################
###################    SEQUENCE IS FOLLWING      ####################################
#
# 1. HANDLING THE DATA REPOSITORY 
# 2. TRAINING 
# 3. TESTING
#
#####################################################################################
#####################################################################################

data = {} 

X_train = np.zeros((266,4,32,32))
X_val   = np.zeros((111,4,32,32))
#y_train = np.zeros((11,))
#y_val   = np.zeros((2,)) 
labeling = np.loadtxt('y_train.txt',unpack=True,dtype='int32')
y_train  = np.transpose(labeling[0:-111])
y_val    = np.transpose(labeling[-111:,])



for i in range(X_train.shape[0]):
	
	img_name = "./dataset_32/train_{}.PNG".format(i+1)           # MAKING TRAINING NAME (NECESSARY !)
	img = sm.imread(img_name)
	a,b,c = img.shape
	img = np.reshape(img,(1,c,a,b))
	X_train[i] = img

for i in range(X_val.shape[0]):
	
	img_name = "./dataset_32/val_{}.PNG".format(i+1)
	img = sm.imread(img_name)
	a,b,c = img.shape
	img = np.reshape(img,(1,c,a,b))
	X_val[i] = img


data['X_train'] = X_train
data['X_val'] = X_val
data['y_train'] = y_train
data['y_val'] = y_val


#########################################################################################
############################# DATA LOADING COMPLETED ####################################
#########################################################################################
 
################################ CHOOSE THE MODEL ! #####################################

#model = KSHNet_BN(input_dim = data['X_train'].shape,hidden_dim=100,num_classes=13
#				,weight_scale=1e-2,reg=5e-3)

model = KSHNet(input_dim=data['X_train'].shape,hidden_dim=100,num_classes=13
				,weight_scale =1e-2,reg=5e-3)

#model = KSHNet(input_dim=data['X_train'].shape,hidden_dim=100,num_classes=13
#				,weight_scale=1e-2,reg=5e-3)

#model = VGGNet(input_dim=data['X_train'].shape,hidden_dim=50,num_classes=13
		       #	,weight_scale=1e-2,reg=5e-3)

#model = ThreeLayerConvNet_2(input_dim = data['X_train'].shape,hidden_dim=300,num_classes=13
#                    	        ,weight_scale=1e-2,reg=5e-3)

solver = Solver(model,data,num_epochs=50,batch_size=70,update_rule='adam',
			optim_config = {
				'learning_rate' :5e-4,
					},
				verbose = True, print_every=10)


start_time = time.time()
solver.train()

print("Training is finished.\n")
print("---------%s seconds ------" % (time.time() - start_time))

plt.subplot(2,1,1)
plt.plot(solver.loss_history,'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2,1,2)
plt.plot(solver.train_acc_history,'-o')
plt.plot(solver.val_acc_history,'-o')
plt.legend(['train','val'],loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()





#img = sm.imread('Pizza.png')
#print img.shape
#a,b,c = img.shape
#img = np.reshape(img,(1,c,a,b))     			# image load 

#model = ThreeLayerConvNet(input_dim=img.shape)
#loss,grads = model.loss(img,y)
"""
w_shape = (3,c,3,3)
w = np.linspace(-0.1,0.1,3*c*3*3).reshape(w_shape)
b = np.linspace(-0.1,0.2,3)  		
conv_param = {'stride':1,'pad':1}


dout = np.random.randn(1,3,img.shape[2],img.shape[3])
out,cache = md.conv_forward(img,w,b,conv_param)
dx,dw,db = md.conv_backward(dout,cache)

print 'dx : ',dx 
print 'dw : ',dw
print 'db : ',db



#out,_ = md.conv_forward(img,w,b,conv_param)
#img2 = sm.imrotate(img,180)
#sm.imshow(img2)
#sm.imsave('outfile.png',img2)

"""

