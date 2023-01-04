from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import torch
def _mse(imageA: np.ndarray, imageB: np.ndarray):

	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA- imageB) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(images: list[tuple[torch.Tensor, torch.Tensor]], title):
	
	#s = ssim(imgAnp, imgBnp)
	# setup the figure
	fig = plt.figure(title, figsize=(2, len(images)))
	fig.set_size_inches(10, 5*len(images))
	for i, (imageA, imageB) in enumerate(images):
		imgAnp, imgBnp = imageA.cpu().numpy().transpose((1,2,0)), imageB.cpu().numpy().transpose((1,2,0))
		# compute the mean squared error and structural similarity
		# index for the images
		m = _mse(imgAnp, imgBnp)
		plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, 0))
		# show first image
		ax = fig.add_subplot(len(images), 2, i*2+1)
		plt.imshow(imgAnp)
		plt.axis("off")
		# show the second image
		ax = fig.add_subplot(len(images), 2, i*2+2)
		plt.imshow(imgBnp)
		plt.axis("off")
	# show the images
	plt.show()