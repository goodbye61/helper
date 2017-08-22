import numpy as np


# STARTING AT 10 MARCH 
# PROJECT NAME : HELPER 


def affine_forward(x,w,b):
	out = None
 	out = x.reshape(x.shape[0],w.shape[0]).dot(w) + b 
 	cache = (x,w,b)
  	return out,cache 


def affine_backward(dout,cache):
  	x,w,b = cache 
 	dx,dw,db = None,None,None
	dx = dout.dot(w.T).reshape(x.shape)
	dw = dout.T.dot(x.reshape(x.shape[0],np.product(x.shape[1:]))).T
	db = np.ones(x.shape[0]).dot(dout)

	return dx,dw,db	

def conv_forward(x,w,b,conv_param):

	N,C,H,W = x.shape
	F,_,HH,WW = w.shape
	stride,pad = conv_param['stride'],conv_param['pad']
	
	Hout = (H+2*pad-HH)/stride + 1 
	Wout = (W+2*pad-WW)/stride + 1 

	out = np.zeros((N,F,Hout,Wout))

	x_padded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
	zipped = C*HH*WW
	filter_reshaped = np.reshape(w,(F,zipped)).T
	
###############################################################################
##########################Filter Sliding#######################################
###############################################################################

	for height in range(Hout):
		top = height * stride	
		bottom = top + HH
		for width in range(Wout):
			left = width*stride	
			right = left + WW

			rec = np.reshape(x_padded[:,:,top:bottom,left:right],(N,zipped))
			output = rec.dot(filter_reshaped) + b 

			out[:,:,height,width] = output 

	cache = (x,w,b,conv_param)
	return out,cache




def conv_backward(dout,cache):

	dx,dw,db = None,None,None
	x,w,b,conv_param = cache
	stride,pad = conv_param['stride'],conv_param['pad']
	N,C,H,W = x.shape
	F,_,HH,WW = w.shape 
	_,_,DH,DW = dout.shape
	
	dx = np.zeros(x.shape)
	dw = np.zeros(w.shape)

	zipped = C*HH*WW

	Hout = (H+2*pad-HH)/stride + 1
 	Wout = (W+2*pad-WW)/stride + 1 
	

	w_reshaped = np.reshape(w,(F,zipped))
	x_padded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
	
	dw_reshaped = np.zeros((F,zipped))
	dx_padded = np.zeros((N,C,H+2*pad,W+2*pad))

	for i in range(Hout):
		top = i * stride
		bottom = top + HH
		for j in range(Wout):
			left = j * stride
			right = left + WW 

			dout_ij = dout[:,:,i,j]

			dx_sub = dout_ij.dot(w_reshaped)
			dx_sub_reshaped = dx_sub.reshape(N,C,HH,WW)
			dx_padded[:,:,top:bottom,left:right] += dx_sub_reshaped 
			
			x_sub_reshaped = x_padded[:,:,top:bottom,left:right].reshape(N,zipped)
			dw_reshaped += dout_ij.T.dot(x_sub_reshaped)

	
	dx = dx_padded[:,:,pad:-pad,pad:-pad]
	dw = dw_reshaped.reshape((F,C,HH,WW))
	db = dout.sum(axis=(0,2,3))

	return dx,dw,db



def max_pool_forward(x,pool_param):
	
	N,C,H,W = x.shape
	pool_height ,pool_width,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']

	Hout = ((H-pool_height)/stride) + 1 
	Wout = ((W-pool_height)/stride) + 1 

	out = np.zeros((N,C,Hout,Wout))

	for i in range(Hout):
		top = i*stride 
		bottom = top + pool_height
		for j in range(Wout):
			left = j*stride
			right = left + pool_width 
			
			out[:,:,i,j] = x[:,:,top:bottom,left:right].max(axis=(2,3))

	cache = (x,pool_param)
	return out,cache 


def max_pool_backward(dout,cache):

	x,pool_param = cache 
	N,C,H,W = x.shape
	pool_height,pool_width,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']

	Wout = ((W-pool_width) / stride) + 1 
 	Hout = ((H-pool_height) / stride) + 1 

	dx = np.zeros((N,C,H,W))

	for i in range(Hout):
		top = i * stride 	
		bottom = top + pool_height 
		for j in range(Wout):
			left = j*stride
			right = left + pool_width 
	
			dout_ij = dout[:,:,i,j].reshape(N*C)
			view = x[:,:,top:bottom,left:right].reshape((N*C,pool_height*pool_width))
			dx_view = dx[:,:,top:bottom,left:right].reshape((N*C,pool_height*pool_width)).T
			
			pos = np.argmax(view,axis=1)

			dx_view[pos,range(N*C)] += dout_ij
			
			dx_view_raw =dx_view.T.reshape(N,C,pool_height,pool_width)
			dx[:,:,top:bottom,left:right] += dx_view_raw


				
	return dx



def softmax_loss(x,y):

	eps = 10e-10
	probs = np.exp(x-np.max(x,axis=1,keepdims=True))
	probs /= np.sum(probs,axis=1,keepdims=True)
	N = x.shape[0] 
	loss = -np.sum(np.log(probs[np.arange(N),y]+eps)) / N
	dx = probs.copy()
	dx[np.arange(N),y] -=1
	dx /= N 
	return loss,dx 
	


def relu_forward(x):

	out = None
	
	out = np.maximum(x,0)
	cache = x 
	return out,cache


def relu_backward(dout,cache):


	dx,x = None,cache
	dx   = dout 
	dx[x<=0] = 0

	return dx 

def batchnorm_forward(x,gamma,beta,bn_param):
   
  mode = bn_param['mode']
  eps = bn_param.get('eps',1e-5)
  momentum = bn_param.get('momentum',0.9)

  N,D =x.shape
  running_mean = bn_param.get('running_mean',np.zeros(D,dtype=x.dtype))
  running_var =  bn_param.get('running_var',np.zeros(D,dtype=x.dtype))
  out,cache = None,None
  if mode == 'train':
  	mu = x.mean(axis=0)
  	xc = x-mu
	var = np.mean(xc**2,axis=0)
  	std = np.sqrt(var+eps)
	xn = xc/std
	out = gamma * xn + beta

	cache = (mode,x,gamma,xc,std,xn,out)

	running_mean += momentum
	running_mean += (1-momentum) * mu 
	
	running_var *=momentum
	running_var += (1-momentum) * var
	
  elif mode =='test':
	std = np.sqrt(running_var + eps)
	xn  = ( x-running_mean) /std
	out = gamma * xn +beta
	cache (mode,x,xn,gamma,beta,std)
  else:
	raise ValueError('Invalid forward batchnorm mode "%s"'%mode)


  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var 


  return out,cache 




def batchnorm_backward(dout,cache):

  
  mode = cache[0]
  if mode == 'train':
	mode,x,gamma,xc,std,xn,out = cache
  
  	N = x.shape[0]
	dbeta = dout.sum(axis=0)
	dgamma = np.sum(xn*dout,axis=0)
	dxn = gamma*dout
	dxc = dxn/std
	dstd = -np.sum((dxn*xc) / (std*std),axis=0)
	dvar = 0.5 * dstd / std
	dxc += (2.0/N) * xc * dvar
	dmu = np.sum(dxc,axis=0)
	dx = dxc-dmu/N

  elif mode == 'test':
	mode,x,xn,gamma,beta,std = cache 
	dbeta = dout.sum(axis=0)
	dgamma = np.sum(xn*dout,axis=0)
	dxn = gamma * dout
 	dx = dxn/std
  else:
	raise ValueError(mode)


  return dx,dgamma,dbeta 



def spatial_batchnorm_forward(x,gamma,beta,bn_param):

  N,C,H,W = x.shape
  x_flat = x.transpose(0,2,3,1).reshape(-1,C)
  out_flat,cache = batchnorm_forward(x_flat,gamma,beta,bn_param)
  out = out_flat.reshape(N,H,W,C).transpose(0,3,1,2)
  
  return out,cache



def spatial_batchnorm_backward(dout,cache):


  N,C,H,W = dout.shape
  dout_flat = dout.transpose(0,2,3,1).reshape(-1,C)
  dx_flat,dgamma,dbeta = batchnorm_backward(dout_flat,cache)
  dx = dx_flat.reshape(N,H,W,C).transpose(0,3,1,2)
  
  return dx,dgamma,dbeta
























 








 









