from module import *


class ThreeLayerConvNet_2(object):
 
  """ 

  Overall architecture :
  
      (spatial batch-norm) - conv - relu - 2x2 max pool - affine - relu - affine - softmax 
        
  """

  def __init__(self,input_dim,num_filters=200,filter_size=3,
		hidden_dim=0,num_classes=10,weight_scale=1e-3,reg=0.0,
		bn_param={'mode':'train'},dtype=np.float32):


	# I have to input the variables, 'Input_dim' !! 	


	self.params = {}
	self.reg = reg
	self.dtype = dtype
	self.bn_param = bn_param

	# Should be Initialized 

	N,C,H,W = input_dim
	HH = filter_size	
	WW = filter_size
	stride = 1
	pad = (filter_size -1 ) /2 	# (7-1)/2 = 3 
	H2 = (1+(H+2*pad-HH)/stride) /2  
	W2 = (1+(W+2*pad-WW)/stride) /2  # -1 original

	#  /2 << Because of 'max_pooling_layer' 

	self.params['W1'] = np.random.normal(0,weight_scale,(num_filters,C,HH,WW))
	self.params['b1'] = np.zeros(num_filters)
	self.params['W2'] = np.random.normal(0,weight_scale,(H2*W2*num_filters,hidden_dim))
	self.params['b2'] = np.zeros(hidden_dim)
	self.params['W3'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
	self.params['b3'] = np.zeros(num_classes)
	self.params['gamma'] = np.random.randn(C)
	self.params['beta'] = np.random.randn(C)

	for k,v in self.params.iteritems():
		self.params[k] = v.astype(dtype)


  def loss(self,X,y=None):

	
	W1,b1 = self.params['W1'],self.params['b1']
	W2,b2 = self.params['W2'],self.params['b2']
	W3,b3 = self.params['W3'],self.params['b3']
	gamma = self.params['gamma']
	beta  = self.params['beta'] 	
	bn_param = self.bn_param	

	filter_size = W1.shape[2]
	pool_param = {'pool_height':2,'pool_width':2,'stride':2}
	conv_param = {'stride':1,'pad':(filter_size-1)/2}
	scores = None
	
	X,cache = spatial_batchnorm_forward(X,gamma,beta,bn_param) 	# SPATIAL B-N
	out1,cache1 = conv_forward(X,W1,b1,conv_param) 		# W1,b1
	out2,cache2 = relu_forward(out1)
	out3,cache3 = max_pool_forward(out2,pool_param)
	out4,cache4 = affine_forward(out3,W2,b2)			# W2,b2
	out5,cache5 = relu_forward(out4)
	out6,cache6 = affine_forward(out5,W3,b3)			# W3,b3

	scores = out6						# get scores.


	if y is None:
		return scores 


	loss,grads = 0,{}
	#print "################## SCORES #######################"
	#print scores 
	#print scores.shape

	softmax_out,dout = softmax_loss(scores,y)
	reg_W1_loss,dW1_reg = self.regularization_loss(self.params['W1'])
	reg_W2_loss,dW2_reg = self.regularization_loss(self.params['W2'])
	reg_W3_loss,dW3_reg = self.regularization_loss(self.params['W3'])
	loss = softmax_out + reg_W1_loss + reg_W2_loss + reg_W3_loss 

	#. Compute the gradient
 	#. Use the module 

	dx3,dW3,db3 = affine_backward(dout,cache6)
	d_temp = relu_backward(dx3,cache5)
	dx2,dW2,db2 = affine_backward(d_temp,cache4)
	
	d_tmp = max_pool_backward(dx2,cache3)
	d_tmp2 = relu_backward(d_tmp,cache2)
	dx1,dW1,db1 = conv_backward(d_tmp2,cache1)
	dx,dgamma,dbeta = spatial_batchnorm_backward(dx1,cache)

	grads['W1'] = dW1_reg + dW1 
 	grads['W2'] = dW2_reg + dW2
	grads['W3'] = dW3_reg + dW3
	grads['b1'] = db1
	grads['b2'] = db2
	grads['b3'] = db3 
	grads['gamma'] = dgamma
	grads['beta'] = dbeta 
	
	
	
	return loss,grads
	
	

  def regularization_loss(self,W):

  	loss = self.reg * 0.5 * np.sum(W*W)
  	d_loss = self.reg * W
  	return loss,d_loss 







 	   
	







