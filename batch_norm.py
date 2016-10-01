import tensortflow as tf

# if x is my input vector where x=<x1,x2,....xn>
# then xnew(i) = (x(i) - mean(x))/sqrt(Var(x)+epsilon)
# return y = gamma * xnew + beta  

def ConvolutionalBatchNormalizer(object):
	def __init__(self,depth,epsilon,ewma,scale_after_norm):
		self.mean = tf.Variable(tf.constant(0.0,shape=[depth]),trainable=False)
		self.variance = tf.Variable(tf.constant(1.0,shape=[depth]),trainable=False)
		self.beta = tf.Variable(tf.constant(0.0,shape=[depth]))
		self.gamma = tf.Variable(tf.constant(1.0,shape=[depth]))
		self.ewma_trainer = ewma
		self.epsilon = epsilon
		self.scale_after_norm = scale_after_norm

# create shadow variable of mean and variance 
	def get_assigner(self):
		return self.ewma_trainer.apply([self.mean,self.variance])


# in batch_norm,
# if phase_train is true
#	calculates mean and variance across depth and stores those values in self.mean and self.variance
#	returns batch_normalized vector

# if phase_train is false
#	mean, variance = stored value of mean , variance from previous iteration
#   local_beta , local_gamma are tensors of same shape as self.beta and self.gamma respectively
#	To avoid training beta and gamma actual tensors are not sent
#	returns batch_normalized vector

	def batch_norm(self,x,phase_train=True):
		if phase_train:
			mean,variance = tf.nn.moments(x,[0,1,2])
			assign_mean = self.mean.assign(mean)
			assign_variance = self.variance.assign(variance)
			with tf.control_dependencies([assign_mean,assign_variance]):
				return tf.nn.batch_normalization(x,mean,variance,self.beta,self.gamma if self.scale_after_norm
											else None,self.epsilon)

		else :
			mean = self.ewma_trainer.average(self.mean)
			variance = self.ewma_trainer.average(self.variance)
			local_beta = tf.identity(self.beta)
			local_gamma = tf.identity(self.gamma)
			return tf.nn.batch_normalization(x,mean,variance,local_beta,local_gamma if self.scale_after_norm
											else None,self.epsilon)
