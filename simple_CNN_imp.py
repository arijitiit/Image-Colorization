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
	img = Image.open(filename)
	rsize = ImageOps.fit(img,(224,224)) 
	rsizeArr = np.asarray(rsize)  # Get array back
	try:
		grey_image_list.append(black_convert(rsizeArr))
		image_list.append(np.asfarray(rsizeArr))
	except:
		continue
	#print black_convert(rsizeArr)
	if i==50: break	

i=0
validate_list = []
val_rgb_list = []
for filename in glob.glob('imagenet/validate_images/*.JPEG'): 
	img = Image.open(filename)
	rsize = ImageOps.fit(img,(224,224)) 
	rsizeArr = np.asarray(rsize)  # Get array back
	try:
		validate_list.append(black_convert(rsizeArr))
		val_rgb_list.append(np.asfarray(rsizeArr))
	except:
		continue
	
	i=i+1
	if i==50 : break



with tf.variable_scope('colors'):
    # Store layers weight
    weights = {
        # 5x5 conv, 1 input, 3 outputs
        'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 50],mean=0.5)),
        # 5x5 conv, 1 input, 3 outputs
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 50, 25],mean=0.5)),
        # 5x5 conv, 1 input, 3 outputs
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 25, 3],mean=0.5)),
    }


with tf.variable_scope('colors'):
    # Store layers bias
    biases = {
        # bias of size 3 for depth 3
        'bc1': tf.Variable(tf.constant(0.1,shape = [50])),
        # bias of size 3 for depth 3
        'bc2': tf.Variable(tf.constant(0.1,shape = [25])),
        # bias of size 3 for depth 3
        'bc3': tf.Variable(tf.constant(0.1,shape = [3])),
    }


grayscale = 	tf.placeholder(tf.float32,[None,224,224,1])
rgb_image =	tf.placeholder(tf.float32,[None,224,224,3])

_tensors = {
	"grayscale":		grayscale,
	"biases":		biases,
	"weights":		weights
}	


pred =tf.mul(cnn_net(_tensors),255.0)

arg1 = tf.div(rgb_image,255)
arg2 = tf.div(pred,255)
arg3 = tf.sub(tf.Variable(tf.constant(1.0,shape = [50,224,224,3])),arg1)
arg4 = tf.sub(tf.Variable(tf.constant(1.0,shape = [50,224,224,3])),arg2)

loss = tf.reduce_mean(-tf.add(tf.mul(arg1,tf.log(arg2)),tf.mul(arg3,tf.log(arg3))),reduction_indices=0)
optimizer = tf.train.AdamOptimizer()
opt = optimizer.minimize(loss)


init = tf.initialize_all_variables()
# Launch the graph.
sess = tf.Session()
sess.run(init)


#merged = tf.merge_all_summaries()
#writer = tf.train.SummaryWriter("tblog", sess.graph)



for step in xrange(1000):
	try:
		print(step)
		list_of_num = random.sample(range(0, len(grey_image_list)), 50)
		batch_xs = list( np.expand_dims(grey_image_list[i],-1) for i in list_of_num ) 
		batch_ys = list( image_list[i] for i in list_of_num ) 
		sess.run(opt,feed_dict={grayscale:batch_xs,rgb_image:batch_ys})

		
		if(step!=0 and step%100==0):
			list_of_num = random.sample(range(0, len(validate_list)), 50)
			batch_xs = list( np.expand_dims(validate_list[i],-1) for i in list_of_num ) 
			batch_ys = list( val_rgb_list[i] for i in list_of_num ) 
			rgb,loss1 = sess.run([pred,loss],feed_dict={grayscale:batch_xs,rgb_image:batch_ys})
			print rgb[0]
		       # sys.stdout.flush()
		       # writer.add_summary(mergedp, step)
        	       # writer.flush()

			for j in range(50):			
				plt.imsave("Output_results_ANN/image_" + str(step) + "_"+str(j), rgb[j])
	except:
		print "Unexpected error:", sys.exc_info()
		break

	
sess.close()

