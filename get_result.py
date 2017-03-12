import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net('/home/deeplearning/work/py-faster-rcnn/caffe-fast-rcnn/models/bvlc_alexnet/deploy.prototxt',
                '/home/deeplearning/work/py-faster-rcnn/VO/dataset/bvlc_alexnet.caffemodel',
                caffe.TEST)

########## for data_t #######################################
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/home/deeplearning/work/py-faster-rcnn/caffe-fast-rcnn/data/ilsvrc12/imagenet_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0)) 
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)


#load the image in the data layer
f = open('test_images.txt', 'r')

a = []
count =0

with open("features_images.txt", "a") as text_file:
	for image_name in f:
		image_name = image_name.strip()
	
		im = caffe.io.load_image(image_name)
		net.blobs['data'].data[...] = transformer.preprocess('data', im)


	#compute
		out = net.forward()
		a.append(out['fc7'])

		count+=1
		#predicted predicted 
		print out['fc7']
		np.savetxt(text_file,out['fc7'],delimiter=',',fmt='%.18e')
	

