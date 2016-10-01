from scipy import misc
import numpy as np
import matplotlib.pyplot as plt # import


def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def black_convert(image):
	#image = misc.imread(string)
	grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
	# get row number

	for rownum in range(len(image)):
	    for column in range(len(image[rownum])):
	        grey[rownum][column] = weightedAverage(image[rownum][column])
	
	#plt.imshow(image, cmap = plt.get_cmap('gray'))
	#plt.show()
	return grey