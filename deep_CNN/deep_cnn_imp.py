import tensorflow as tf
#import utils
import glob
from batch_norm import ConvolutionalBatchNormalizer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

filenames = glob.glob("imagenet/Imagenet_dataset/*")
num_epochs = 1e9
batch_size = 1

phase_train = tf.placeholder(tf.bool,name='phase_train')


def read_file(filename_queue):
	reader = tf.WholeFileReader()
	key,content = reader.read(filename_queue)
	image = tf.image.decode_jpeg(content,channels=3)
	image = tf.image.resize_images(image,[224,224])
	float_img = tf.div(tf.cast(image,tf.float32),255)
	return float_img


def input_pipeline(filenames):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs)
	exp = read_file(filename_queue)
	capacity = 200
	min_after_dequeue = 100
	return tf.train.shuffle_batch([exp], batch_size = batch_size,
      capacity = capacity, min_after_dequeue = min_after_dequeue)


def batch_normalize(x,depth,phase_train):
	ewma = tf.train.ExponentialMovingAverage(decay=0.9999)
	epsilon = 1e-3
	obj = 	ConvolutionalBatchNormalizer(depth,epsilon,ewma,True)
	check = obj.get_assigner()
	return tf.cond(phase_train,lambda: obj.batch_norm(x,True),lambda: obj.batch_norm(x,False))



def conv2d(inp,filter,phase_train):
	conv_out = tf.nn.conv2d(inp,filter,[1,1,1,1],padding = 'SAME')
	normed_out = batch_normalize(conv_out,filter.get_shape()[3],phase_train)
	return tf.nn.relu(normed_out)



def rgb_to_yuv(img):
	#print img.get_shape()
	weights = tf.constant([[[[0.299,-0.169,0.499],[0.587,-0.331,-0.418],[0.114,0.499,-0.0813]]]])
	#print weights.get_shape()
	biases = tf.constant([0.,0.5,0.5])
	conv_out = tf.nn.conv2d(img,weights,[1,1,1,1],padding = 'SAME')
	return tf.nn.bias_add(conv_out,biases)


def yuv_to_rgb(img):
	temp_img = tf.mul(img,255)
	weights = tf.constant([[[[1.0,1.0,1.0],[0.0,-0.344,1.772],[1.402,-0.714,0.0]]]])
	biases = tf.constant([-179.456,135.424,-226.816])
	conv_out = tf.nn.conv2d(temp_img,weights,[1,1,1,1],padding='SAME')
	out = tf.nn.bias_add(conv_out,biases)	

	#clamp between 0 and 255
	max = tf.constant(255.0,shape=[batch_size,224,224,3])
	min = tf.constant(0.0,shape=[batch_size,224,224,3])
	temp = tf.minimum(max,out)
	return tf.maximum(min,temp)
	



def build_cnn(tensors):
	# first conv_layer 28*28*512 ---> 28*28*256
	layer1 = conv2d(batch_normalize(tensors["conv4_3"],512,phase_train),tensors["weights"]["wc1"],phase_train)	
	resized_layer1 = tf.image.resize_bilinear(layer1,[56,56])
	input1 = tf.add(resized_layer1,batch_normalize(tensors["conv3_3"],256,phase_train))
	
	# second conv_layer 56*56*256 ---> 56*56*128
	layer2 = conv2d(input1,tensors["weights"]["wc2"],phase_train)
	resized_layer2 = tf.image.resize_bilinear(layer2,[112,112])
	input2 = tf.add(resized_layer2,batch_normalize(tensors["conv2_2"],128,phase_train))	
	
	# third conv_layer 112*112*128 ---->112*112*64
	layer3 = conv2d(input2,tensors["weights"]["wc3"],phase_train)
	resized_layer3 = tf.image.resize_bilinear(layer3,[224,224])
	input3 = tf.add(resized_layer3,batch_normalize(tensors["conv1_2"],64,phase_train))

	# fourth conv_layer 224*224*64 ----> 224*224*3
	layer4 = conv2d(input3,tensors["weights"]["wc4"],phase_train)
	input4 = tf.add(layer4,batch_normalize(tensors["grayscale"],3,phase_train))

	# fifth conv_layer  224*224*3 ----> 224*224*3
	layer5 = conv2d(input4,tensors["weights"]["wc5"],phase_train)
	
	# sixth conv_layer  224*224*3 ----> 224*224*2
	layer6 = conv2d(layer5,tensors["weights"]["wc6"],phase_train)

	return tf.sigmoid(layer6)










colorimage = input_pipeline(filenames)
#print colorimage.get_shape()
grayscale = tf.image.rgb_to_grayscale(colorimage)
colorimage_yuv = rgb_to_yuv(colorimage)

colorimage_uv = tf.concat(3,[tf.split(3,3,colorimage_yuv)[1],tf.split(3,3,colorimage_yuv)[2]])

grayscale = tf.concat(3,[grayscale,grayscale,grayscale])




with open("vgg16.tfmodel", mode='rb') as f:
	fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def, input_map={ "images": grayscale })
graph = tf.get_default_graph()


with tf.variable_scope('vgg'):
	# intermediate output layers of vgg-16
	conv1_2 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
	conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
	conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
	conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")

with tf.variable_scope('colors'):
    # Store layers weight
    weights = {
        # 1x1 conv, 512 input_ch, 256 output_ch
        'wc1': tf.Variable(tf.truncated_normal([1, 1, 512, 256],mean=0.0)),
        # 3x3 conv, 256 input_ch, 128 output_ch
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128],mean=0.0)),
        # 3x3 conv, 128 input_ch, 64 output_ch
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64],mean=0.0)),
		# 3x3 conv, 64 input_ch, 3 output_ch
        'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 3],mean=0.0)),
		# 3x3 conv, 3 input_ch, 3 output_ch
        'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 3],mean=0.0)),
		# 3x3 conv, 3 input_ch, 3 output_ch
        'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 2],mean=0.0)),
    }



_tensors = {
	"conv1_2":		conv1_2,
	"conv2_2":		conv2_2,
	"conv3_3":		conv3_3,
	"conv4_3":		conv4_3,
	"grayscale":	grayscale,
	"weights":		weights
}




# build the cnn network
output_uv = build_cnn(_tensors)
output_yuv = tf.concat(3,[tf.split(3,3,grayscale)[0],output_uv])
print tf.split(3,3,grayscale)[0].get_shape()
output_rgb = yuv_to_rgb(output_yuv)
print output_rgb.get_shape(),colorimage.get_shape()
output_image = tf.concat(2,[output_rgb,colorimage])


# calculate loss function
loss = tf.reduce_mean(tf.square(tf.sub(output_uv,colorimage_uv)),reduction_indices = 0)

# define optimizer learning rate 1e-3
optimizer = tf.train.AdamOptimizer()
opt = optimizer.minimize(loss)


init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
# Launch the graph.
sess = tf.Session()
sess.run(init_op)


# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

step = 0
try:
  while not coord.should_stop():
    #...do some work...
	print step
	step=step+1
	sess.run(opt,feed_dict={phase_train : False})
	if step%1000 == 0:
		compare_output = sess.run(output_image,feed_dict={phase_train : False})
		plt.imsave("Output_results_ANN/image_" + str(step) , compare_output[0])
		

except Exception as e:
  coord.request_stop(e)



coord.join(threads)
sess.close()








