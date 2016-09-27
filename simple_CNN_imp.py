from PIL import Image
from PIL import ImageOps
import glob
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from black_convert import black_convert
import random
import tensorflow as tf

#global_step = tf.Variable(0, name='global_step', trainable=False)

def conv2d(_X, w):
    with tf.variable_scope('conv2d'):
       _X = tf.nn.conv2d(_X, w , [1, 1, 1, 1], padding = 'SAME')
       return _X





# main CNN network used for training
def cnn_net(_tensors):
	with tf.variable_scope('colors'):
		conv1 = tf.sigmoid(conv2d(_tensors["grayscale"],_tensors["weights"]["wc1"])+_tensors["biases"]["bc1"])		
		conv2 = tf.sigmoid(conv2d(conv1,_tensors["weights"]["wc2"])+_tensors["biases"]["bc2"])
		conv3 = tf.sigmoid(conv2d(conv2,_tensors["weights"]["wc3"])+_tensors["biases"]["bc3"])
		return conv3




image_list = []
grey_image_list = []
i=0
for filename in glob.glob('imagenet/Imagenet_dataset/*.JPEG'): 
	print i
	i = i+1
	try:
		img = Image.open(filename)
		rsize = ImageOps.fit(img,(224,224)) 
		ones = np.ones((224,224,3))
		onesp = np.ones((224,224))
		rsizeArr = np.asarray(rsize)  # Get array back
#		print 'train'
#		print rsizeArr[0][0]
		grey_image_list.append(onesp-np.divide(black_convert(rsizeArr),255))
		image_list.append(ones-np.divide(np.asfarray(rsizeArr),255))
	except:
		continue
	#print black_convert(rsizeArr)
	#if i==50: break	
#print 'grey'
#grey_image_list[0][0][0]


i=0
validate_list = []
val_rgb_list = []
for filename in glob.glob('imagenet/validate_images/*.JPEG'): 
	try:
		img = Image.open(filename)
		rsize = ImageOps.fit(img,(224,224))
		ones = np.ones((224,224,3))
        	onesp = np.ones((224,224))
 #		print 'val'
		rsizeArr = np.asarray(rsize)  # Get array back
#		print rsizeArr[0][0]
		validate_list.append(onesp-np.divide(black_convert(rsizeArr),255))
		val_rgb_list.append(ones-np.divide(np.asfarray(rsizeArr),255))
	except:
		continue
	print i
	i=i+1
	#if i==51: break

print 'validate'
print validate_list[0][0][0]


with tf.variable_scope('colors'):
    # Store layers weight
    weights = {
        # 5x5 conv, 1 input, 3 outputs
        'wc1': tf.Variable(tf.truncated_normal([1, 1, 1, 50],stddev=0.01)),
        # 5x5 conv, 1 input, 3 outputs
        'wc2': tf.Variable(tf.truncated_normal([1, 1, 50, 25],stddev=0.01)),
        # 5x5 conv, 1 input, 3 outputs
        'wc3': tf.Variable(tf.truncated_normal([1, 1, 25, 3],stddev=0.01)),
    }


with tf.variable_scope('colors'):
    # Store layers bias
    biases = {
        # bias of size 3 for depth 3
        'bc1': tf.Variable(tf.constant(0.0,shape = [50])),
        # bias of size 3 for depth 3
        'bc2': tf.Variable(tf.constant(0.0,shape = [25])),
        # bias of size 3 for depth 3
        'bc3': tf.Variable(tf.constant(0.0,shape = [3])),
    }


grayscale = 	tf.placeholder(tf.float32,[None,224,224,1])
rgb_image =	tf.placeholder(tf.float32,[None,224,224,3])

_tensors = {
	"grayscale":		grayscale,
	"biases":		biases,
	"weights":		weights
}	


pred = cnn_net(_tensors)
image = tf.concat(2,[pred,rgb_image])
#arg1 = tf.div(tf.add(tf.Variable(tf.constant(1.0,shape = [50,224,224,3])),rgb_image),257)
#arg2 = tf.div(tf.add(tf.Variable(tf.constant(1.0,shape = [50,224,224,3])),pred),257)
arg3 = tf.sub(tf.Variable(tf.constant(1.0,shape = [50,224,224,3])),pred)
arg4 = tf.sub(tf.Variable(tf.constant(1.0,shape = [50,224,224,3])),rgb_image)

arg1 = tf.clip_by_value(arg3,1e-10,1)
arg2 = tf.clip_by_value(pred,1e-10,1)


loss = tf.reduce_sum(-tf.reduce_mean(tf.add(tf.mul(rgb_image,tf.log(arg2)),tf.mul(arg4,tf.log(arg1))),reduction_indices=0))
optimizer = tf.train.AdamOptimizer(0.0001)
opt = optimizer.minimize(loss)


init = tf.initialize_all_variables()
# Launch the graph.
sess = tf.Session()
sess.run(init)


#merged = tf.merge_all_summaries()
#writer = tf.train.SummaryWriter("tblog", sess.graph)

#minloss = 1e15

for step in xrange(1000000):
	try:
		print(step)
		list_of_num = random.sample(range(0, len(grey_image_list)), 50)
		batch_xs = list( np.expand_dims(grey_image_list[i],-1) for i in list_of_num ) 
		batch_ys = list( image_list[i] for i in list_of_num ) 
		sess.run(opt,feed_dict={grayscale:batch_xs,rgb_image:batch_ys})
		loss2 = sess.run(loss,feed_dict = {grayscale:batch_xs,rgb_image:batch_ys})
		#print loss2,imag[0][0][0]
		print loss2
		if  loss2<=1e1:
			break

		
		if(step!=0 and step%1000==0):
			#print len(validate_list)
			list_of_num = random.sample(range(0, len(validate_list)),50)
			batch_xs = list( np.expand_dims(validate_list[i],-1) for i in list_of_num ) 
			batch_ys = list( val_rgb_list[i] for i in list_of_num ) 
			rgb = sess.run([image],feed_dict = {grayscale:batch_xs,rgb_image:batch_ys})[0]
			#print rgb[0]
			#print loss1

			#rgb[rgb<0] = 0
			#rgb[rgb>1] = 1
			rgb = np.multiply(rgb,255.0)
		       # sys.stdout.flush()
		       # writer.add_summary(mergedp, step)
        	       # writer.flush()

			for j in range(50):			
				plt.imsave("Output_results_ANN/image_" + str(step) + "_"+str(j), rgb[j],vmin=0,vmax=255)
			#if loss1>=minloss or loss1<=1e1:
			#	break
			#else:
			#	minloss = loss1
	except:
		print "Unexpected error:", sys.exc_info()
		break

	
sess.close()

