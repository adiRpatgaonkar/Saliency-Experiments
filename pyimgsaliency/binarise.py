import numpy as np
np.set_printoptions(threshold=np.nan)

def binarise_saliency_map(saliency_map,method='adaptive', threshold=0.5, adaptive_factor=2.0):

	# check if input is a numpy array
	if type(saliency_map).__module__ != np.__name__:
		print('Expected numpy array')
		return None

	#check if input is 2D
	if len(saliency_map.shape) != 2:
		print('Saliency map must be 2D')
		return None

	if method == 'fixed':
		saliency_map = saliency_map / 255.0
		return np.where(saliency_map > threshold, 1, 0)

	elif method == 'adaptive':
		adaptive_threshold = adaptive_factor * saliency_map.mean()
		saliency_map = saliency_map / 255.0
		#print(saliency_map)
		return np.where(saliency_map > adaptive_threshold, 1, 0)

	elif method == 'clustering':
		print('Not yet implemented')
		return None

	else:
		print("Method must be one of fixed, adaptive or clustering")
		return None
