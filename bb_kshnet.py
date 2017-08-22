from module import *


class KSHNet_BN(object):
 
  """ 

  Overall architecture :
  
 
   conv - relu - conv - relu - max pooling - conv - relu - max pooiling 
			- FC - relu - FC - SOFTMAX 


  """

  def __init__(self,input_dim,num1_filters=96,filter1_size=3,
		num2_filters=32,filter2_size=3,
		num3_filters=0,filter3_size=3,
		hidden_dim=100,num_classes=10,weight_scale=1e-3,reg=0.0,
		bn_param={'mode':'train'},
		dtype=np.float32):


	# I have to input the variables, 'Input_dim' !! 	


	self.params = {}
	self.reg = reg
	self.dtype = dtype
 	self.bn_param = bn_param
	# INITIALIZING FILTER PARAMETER 

	N,C,H,W = input_dim
	H1 = filter1_size	
	W1 = filter1_size
	H2 = filter2_size
	W2 = filter2_size
	H3 = filter3_size
	W3 = filter3_size
	stride = 1

	pad1 = (filter1_size -1)/2
	pad2 = (filter2_size -1)/2
 	pad3 = (filter3_size -1)/2
	O1_H = (1+(H+2*pad1-H1)/stride) 	# OUTPUT 1 
	O1_W = (1+(W+2*pad1-W1)/stride)         # OUTPUT 2
						# PASSING THE CONV_1 LAYER 
	O2_H = (1+(O1_H+2*pad2-H2)/stride)/2
	O2_W = (1+(O1_W+2*pad2-W2)/stride)/2
						# PASSING THE CONV_2 & MAX POOLING LAYER
	O3_H = (1+(O2_H+2*pad3-H3)/stride)/2
	O3_W = (1+(O2_W+2*pad3-W3)/stride)/2    # PASSING THE CONV_3 & MAXPOOLING LAYER 
	 	
	
	self.params['W1'] = np.random.normal(0,weight_scale,(num1_filters,C,H1,W1))
	self.params['b1'] = np.zeros(num1_filters)
	self.params['W2'] = np.random.normal(0,weight_scale,(num2_filters,num1_filters,H2,W2))
	self.params['b2'] = np.zeros(num2_filters)
	#self.params['W3'] = np.random.normal(0,weight_scale,(num3_filters,num2_filters,H3,W3))
	#self.params['b3'] = np.zeros(num3_filters)
	self.params['W4'] = np.random.normal(0,weight_scale,(O2_H*O2_W*num2_filters,hidden_dim))
	self.params['b4'] = np.zeros(hidden_dim)
	self.params['W5'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
	self.params['b5'] = np.zeros(num_classes)

	self.params['gamma'] = np.random.randn(num1_filters)
	self.params['beta'] = np.random.randn(num1_filters)

	for k,v in self.params.iteritems():
		self.params[k] = v.astype(dtype)


  def loss(self,X,y=None):
	
	W1,b1 = self.params['W1'],self.params['b1']
	W2,b2 = self.params['W2'],self.params['b2']
	#W3,b3 = self.params['W3'],self.params['b3']
	W4,b4 = self.params['W4'],self.params['b4']
	W5,b5 = self.params['W5'],self.params['b5']
 	
	gamma = self.params['gamma']
	beta  = self.params['beta']
  	bn_param = self.bn_param


	filter1_size = W1.shape[2]
	filter2_size = W2.shape[2]
	#filter3_size = W3.shape[2]
	
	pool_param = {'pool_height':2,'pool_width':2,'stride':2}
	conv1_param = {'stride':1,'pad':(filter1_size-1)/2}
	conv2_param = {'stride':1,'pad':(filter2_size-1)/2}
	#conv3_param = {'stride':1,'pad':(filter3_size-1)/2}

	scores = None

	#out0,cache  = spatial_batchnorm_forward(X,gamma,beta,bn_param)
	out1,cache1 = conv_forward(X,W1,b1,conv1_param) 			 # PASSING CONV1_NET 
	out_1,cache_1 = spatial_batchnorm_forward(out1,gamma,beta,bn_param)
        out2,cache2 = relu_forward(out_1)
	out3,cache3 = conv_forward(out2,W2,b2,conv2_param)		 # PASSING CONV2_NET 
	out4,cache4 = relu_forward(out3)
	out5,cache5 = max_pool_forward(out4,pool_param)			 
	#out6,cache6 = conv_forward(out5,W3,b3,conv3_param)		 # PASSING CONV3_NET 
	#out7,cache7 = relu_forward(out6)				 # PASSING RELU 
	#out8,cache8 = max_pool_forward(out7,pool_param)		         # PASSING MAX-POOLING 

	out9,cache9 = affine_forward(out5,W4,b4)
	out10,cache10 = relu_forward(out9)
	out11,cache11 = affine_forward(out10,W5,b5)

	scores = out11 


	if y is None:
		return scores 


	# COMPUTING LOSS 	

	loss,grads = 0,{}

	softmax_out,dout    = softmax_loss(scores,y)
	reg_W1_loss,dW1_reg = self.regularization_loss(self.params['W1'])
	reg_W2_loss,dW2_reg = self.regularization_loss(self.params['W2'])
	#reg_W3_loss,dW3_reg = self.regularization_loss(self.params['W3'])
	reg_W4_loss,dW4_reg = self.regularization_loss(self.params['W4'])
	reg_W5_loss,dW5_reg = self.regularization_loss(self.params['W5'])
		
	loss = reg_W2_loss + softmax_out + reg_W1_loss + reg_W4_loss + reg_W5_loss 

	dx5,dW5,db5 = affine_backward(dout,cache11)
	dx_ = relu_backward(dx5,cache10)
	dx4,dW4,db4 = affine_backward(dx_,cache9)
	#dx_1 = max_pool_backward(dx4,cache8)
	#dx_2 = relu_backward(dx_1,cache7)
	#dx3,dW3,db3 = conv_backward(dx_2,cache6)
	dx_  = max_pool_backward(dx4,cache5)
	dx_1 = relu_backward(dx_,cache4)
	dx2,dW2,db2 = conv_backward(dx_1,cache3)
	dx__  = relu_backward(dx2,cache2)
        dx_,dgamma,dbeta   = spatial_batchnorm_backward(dx__,cache_1)
	dx1,dW1,db1 = conv_backward(dx__,cache1)	
	#dx,dgamma,dbeta = spatial_batchnorm_backward(dx1,cache)	

	grads['W1'] = dW1_reg + dW1 
 	grads['W2'] = dW2_reg + dW2
	#grads['W3'] = dW3_reg + dW3
	grads['W4'] = dW4_reg + dW4
	grads['W5'] = dW5_reg + dW5
	grads['b1'] = db1
	grads['b2'] = db2
	#grads['b3'] = db3 
	grads['b4'] = db4
	grads['b5'] = db5
	grads['gamma'] = dgamma
	grads['beta'] = dbeta
	
	
	return loss,grads
	
	

  def regularization_loss(self,W):

  	loss = self.reg * 0.5 * np.sum(W*W)
  	d_loss = self.reg * W
  	return loss,d_loss 







 	   
	







