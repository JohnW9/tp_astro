#cython: language_level=3
import numpy as np
import miscmath as mm
import copy
from scipy import interpolate
from skimage.measure import label, regionprops
from skimage.filters import gaussian as gaussian_filter
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
import DEFINES
import time

#Get the exact location of the centroid
def compute_centroid(rawImage, cameraProps, result_ID):
	"""
	Extracts the centroid of an image

	It will take the raw image, filter it, subtract the camera distortion and find the centroid in it.
	Only one centroid will be recovered. The centroid location and shape are returned.

	Parameters
	----------
	rawImage: np.ndarray
		The image containing the centroid.
	cameraProps: calssCamera.CameraParameters
		The camera properties at the time the image was taken. If the image was altered (for instance, cropped),
		the ROIOffsetX and ROIOffsetY parameters must be adapted. The properties must also contain the 
		camera distortion data as well as the camera type.
	result_ID: int
		The unique identifier for the image

	Returns
	-------
	Tuple: (X (float), Y (float), X-px (float), Y-px (float), sigmaX (float), sigmaY (float), height (float), resultID (int))
		X: The X coordinate of the centroid in millimeters\n
		Y: The Y coordinate of the centroid in millimeters\n
		X: The X coordinate of the centroid in pixels\n
		Y: The Y coordinate of the centroid in pixels\n
		sigmaX: The centroid width in the X drection, in pixels\n
		sigmaY: The centroid width in the Y drection, in pixels\n
		height: The height of the fitted gaussian. Representative of the centroid's brightness.\n
		resultID: The image unique identifier
	"""

	col_corr = cameraProps.yCorr
	row_corr = cameraProps.xCorr
	# col_corr = cameraProps.xCorr
	# row_corr = cameraProps.yCorr
	# col_corr = np.multiply(cameraProps.xCorr,0)
	# row_corr = np.multiply(cameraProps.yCorr,0)
	col_offset = cameraProps.ROIoffsetX
	row_offset = cameraProps.ROIoffsetY
	scale_factor = cameraProps.scaleFactor
	test_bench = cameraProps.cameraType
	image_shape = rawImage.shape
	col_corr = col_corr[(row_offset):(row_offset+image_shape[DEFINES.CC_ROW_COORDINATE]),\
						(col_offset):(col_offset+image_shape[DEFINES.CC_COL_COORDINATE])]
	row_corr = row_corr[(row_offset):(row_offset+image_shape[DEFINES.CC_ROW_COORDINATE]),\
						(col_offset):(col_offset+image_shape[DEFINES.CC_COL_COORDINATE])]
	
	# Filter the 2D image
	rawImage = np.divide(rawImage, DEFINES.PC_CAMERA_MAX_INTENSITY_RAW).astype(np.float_)
	rawImage = np.nan_to_num(rawImage)
	imgSave = copy.deepcopy(rawImage)
	maxIntensity = np.max(gaussian_filter(rawImage, DEFINES.CC_IMAGE_XY_FILTERING_SIGMA))
	# rawImage = denoise_wavelet(rawImage, sigma = 3, wavelet='coif5', mode='soft', wavelet_levels=None, method='VisuShrink', rescale_sigma = True)
	# rawImage = gaussian_filter(rawImage, DEFINES.CC_IMAGE_XY_FILTERING_SIGMA)
	filteredImage = rawImage.astype(np.longdouble)
	
	# lalrow = np.array([np.mgrid[0:image_shape[DEFINES.CC_ROW_COORDINATE]],]*(image_shape[DEFINES.CC_COL_COORDINATE])).transpose()
	# lalcol = np.array([np.mgrid[0:image_shape[DEFINES.CC_COL_COORDINATE]],]*(image_shape[DEFINES.CC_ROW_COORDINATE]))
	
	# print([row_offset, row_offset+image_shape[DEFINES.CC_ROW_COORDINATE], col_offset, col_offset+image_shape[DEFINES.CC_COL_COORDINATE]])
	# if image_shape[0] < 500:
	# 	fig = plt.figure(figsize=plt.figaspect(0.333333333333333))
	# 	ax = fig.add_subplot(1, 3, 1, projection='3d')
	# 	#pcolormesh
	# 	ax.plot_surface(lalcol, lalrow, imgSave, cmap=plt.cm.viridis)
	# 	ax = fig.add_subplot(1, 3, 2, projection='3d')
	# 	ax.plot_surface(lalcol, lalrow, filteredImage, cmap=plt.cm.viridis)
	# 	ax = fig.add_subplot(1, 3, 3, projection='3d')
	# 	ax.plot_surface(lalcol, lalrow, np.subtract(filteredImage,imgSave), cmap=plt.cm.viridis)
	# 	plt.draw()
	# 	plt.pause(1e-17)
	# 	plt.show()

	#Check if the image is sufficiently bright
	if maxIntensity < DEFINES.CC_CENTROID_DETECTION_THRESHOLD:
		return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,result_ID]

	#Check if the image is sufficiently big
	if test_bench == DEFINES.PC_CAMERA_TYPE_XY:
		if image_shape[0] < DEFINES.CC_CENTROID_XY_MAX_DIAMETER*DEFINES.CC_SMALL_IMAGE_CROP_ROW_RATIO_XY*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_XY or image_shape[1] < DEFINES.CC_CENTROID_XY_MAX_DIAMETER*DEFINES.CC_SMALL_IMAGE_CROP_COL_RATIO_XY*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_XY:
			return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,result_ID]
	elif test_bench == DEFINES.PC_CAMERA_TYPE_TILT:
		if image_shape[0] < DEFINES.CC_CENTROID_TILT_MAX_DIAMETER*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_TILT or image_shape[1] < DEFINES.CC_CENTROID_TILT_MAX_DIAMETER*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_TILT:
			return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,result_ID]
	else:
		return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,result_ID]
	#Get connected points of the image
	if test_bench == DEFINES.PC_CAMERA_TYPE_XY:
		label_img = label(filteredImage > DEFINES.CC_CENTROID_DETECTION_THRESHOLD_XY_RATIO*np.max(filteredImage))
	elif test_bench == DEFINES.PC_CAMERA_TYPE_TILT:
		label_img = label(filteredImage > DEFINES.CC_CENTROID_DETECTION_THRESHOLD_TILT_RATIO*np.max(filteredImage))
	else:
		return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,result_ID]
	props = regionprops(label_img, intensity_image=filteredImage)

	if len(props) < 1:
		return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,result_ID]

	#search the first big dot in the properties. This avoids most reflectance artifacts.
	for i in range(0,len(props)+1):
		if i >= len(props):
			# for i in range(0,len(props)):
			# 	log.message(DEFINES.LOG_MESSAGE_PRIORITY_DEBUG_INFO, 1, f'Diameter {props[i].equivalent_diameter:.2f}')
			return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,result_ID]

		diameter  = props[i].equivalent_diameter
		intensity = props[i].max_intensity
		if 	((diameter > DEFINES.CC_CENTROID_XY_MIN_DIAMETER and diameter < DEFINES.CC_CENTROID_XY_MAX_DIAMETER and test_bench == DEFINES.PC_CAMERA_TYPE_XY) or \
			(diameter > DEFINES.CC_CENTROID_TILT_MIN_DIAMETER and diameter < DEFINES.CC_CENTROID_TILT_MAX_DIAMETER and test_bench == DEFINES.PC_CAMERA_TYPE_TILT)) and \
			intensity >= DEFINES.CC_CENTROID_DETECTION_THRESHOLD: #filter diameter
			centroids = props[i].centroid
			if len(centroids) >= 2:
				if centroids[DEFINES.CC_ROW_COORDINATE] >= 0 and centroids[DEFINES.CC_ROW_COORDINATE] <= (filteredImage.shape)[DEFINES.CC_ROW_COORDINATE] \
				and centroids[DEFINES.CC_COL_COORDINATE] >= 0 and centroids[DEFINES.CC_COL_COORDINATE] <= (filteredImage.shape)[DEFINES.CC_COL_COORDINATE]:
					break
	#crop the image
	if test_bench == DEFINES.PC_CAMERA_TYPE_XY:
		crop_size_row_min = -int(diameter*DEFINES.CC_SMALL_IMAGE_CROP_ROW_RATIO_XY*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_XY)
		crop_size_row_max = int(diameter*DEFINES.CC_SMALL_IMAGE_CROP_ROW_RATIO_XY*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_XY)
		crop_size_col_min = -int(diameter*DEFINES.CC_SMALL_IMAGE_CROP_COL_RATIO_XY*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_XY)
		crop_size_col_max = int(diameter*DEFINES.CC_SMALL_IMAGE_CROP_COL_RATIO_XY*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_XY)
	elif test_bench == DEFINES.PC_CAMERA_TYPE_TILT:
		crop_size_row_min = -int(diameter*DEFINES.CC_SMALL_IMAGE_CROP_ROW_RATIO_TILT*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_TILT)
		crop_size_row_max = int(diameter*DEFINES.CC_SMALL_IMAGE_CROP_ROW_RATIO_TILT*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_TILT)
		crop_size_col_min = -int(diameter*DEFINES.CC_SMALL_IMAGE_CROP_COL_RATIO_TILT*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_TILT)
		crop_size_col_max = int(diameter*DEFINES.CC_SMALL_IMAGE_CROP_COL_RATIO_TILT*DEFINES.CC_COMPUTATION_SIGMA_CROP_RATIO_TILT)
	else:
		return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,result_ID]

	row_centroid = int(centroids[DEFINES.CC_ROW_COORDINATE])
	col_centroid = int(centroids[DEFINES.CC_COL_COORDINATE])

	#Check that the crop remains in the image
	if row_centroid+crop_size_row_min<0:
		crop_size_row_min = -row_centroid
	if row_centroid+crop_size_row_max>image_shape[DEFINES.CC_ROW_COORDINATE]:
		crop_size_row_max = image_shape[DEFINES.CC_ROW_COORDINATE]-row_centroid
	if col_centroid+crop_size_col_min<0:
		crop_size_col_min = -col_centroid
	if col_centroid+crop_size_col_max>image_shape[DEFINES.CC_COL_COORDINATE]:
		crop_size_col_max = image_shape[DEFINES.CC_COL_COORDINATE]-col_centroid

	#Take the raw image for the gaussian fitting
	image_small = rawImage[	(row_centroid+crop_size_row_min):(row_centroid+crop_size_row_max),\
							(col_centroid+crop_size_col_min):(col_centroid+crop_size_col_max)]

	col_corr_small = col_corr[	(row_centroid+crop_size_row_min):(row_centroid+crop_size_row_max),\
								(col_centroid+crop_size_col_min):(col_centroid+crop_size_col_max)]
	row_corr_small = row_corr[	(row_centroid+crop_size_row_min):(row_centroid+crop_size_row_max),\
								(col_centroid+crop_size_col_min):(col_centroid+crop_size_col_max)]

	#Apply the correction on the cropped image. Arrays are [ROW,COL] instead of [x,y]
	rowIn = np.array([np.mgrid[0:crop_size_row_max-crop_size_row_min],]*(crop_size_col_max-crop_size_col_min)).transpose()
	colIn = np.array([np.mgrid[0:crop_size_col_max-crop_size_col_min],]*(crop_size_row_max-crop_size_row_min))
	
	# a = interpolate.interp2d(rowIn, colIn, np.add(rowIn,row_corr_small), kind='linear', fill_value = 0)
	# print(a(10,10))

	rowIn = np.add(rowIn, row_corr_small)
	colIn = np.add(colIn, col_corr_small)
	
	data_min = DEFINES.CC_IMAGE_THRESHOLD_MIN
	data_max = DEFINES.CC_IMAGE_THRESHOLD_MAX_RATIO*np.max(image_small)

	mm.threshold(image_small,data_min,data_max)

	current_image = image_small.copy()
	current_colIn = colIn.copy()
	current_rowIn = rowIn.copy()
	
	#Fit a 2D mm.gaussian
	if cameraProps.cameraType == DEFINES.PC_CAMERA_TYPE_XY:
		estimate = mm.moments(current_image,current_colIn,current_rowIn)
		if estimate == None:
			estimate = (intensity, abs(crop_size_col_min), abs(crop_size_row_min), 1/(2*np.sqrt(2*np.log(1/intensity))), 1/(2*np.sqrt(2*np.log(1/intensity))), 1)
		params = mm.fitgaussian(current_image,current_colIn,current_rowIn,data_min,data_max, DEFINES.CC_OPTIMIZER_TOLERANCE_XY, estimate)

	else:
		estimate = mm.moments(current_image,current_colIn,current_rowIn)
		if estimate == None:
			estimate = (intensity, abs(crop_size_col_min), abs(crop_size_row_min), 1/(2*np.sqrt(2*np.log(1/intensity))), 1/(2*np.sqrt(2*np.log(1/intensity))), 1)
		params = mm.fitgaussian(current_image,current_colIn,current_rowIn,data_min,data_max, DEFINES.CC_OPTIMIZER_TOLERANCE_TILT, estimate)
	
	(height, center_col, center_row, width_col, width_row, n, angle) = params

	#rescale the result
	center_col = center_col+crop_size_col_min
	center_row = center_row+crop_size_row_min

	col_out = col_centroid+center_col+col_offset # Add centroid offset and ROI offset
	row_out = row_centroid+center_row+row_offset

	return [col_out*scale_factor,row_out*scale_factor,col_out,row_out,width_col,width_row,height,result_ID]

def _main():
	import pywt
	import classCamera as cam
	from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma,denoise_nl_means,denoise_tv_bregman)
	import cv2
	import copy
	import time

	tStart = time.perf_counter()

	imgSize = DEFINES.PC_CAMERA_XY_CALIB_CROP
	nbImages = 1000
	noiseLevel = 0
	height = 0.9
	centroid = (50,50)
	sx = 1.3
	sy = 1.35
	n = 1.02

	noiseSTD = 1
	heightSTD = 0.03
	centroidSTD = 0.0
	sSTD = 0.05
	nSTD = 0.08

	cameraProps = cam.CameraParameters(DEFINES.PC_CAMERA_TYPE_XY)

	cameraProps.yCorr = np.zeros((imgSize, imgSize))
	cameraProps.xCorr = np.zeros((imgSize, imgSize))
	cameraProps.ROIoffsetX = 0
	cameraProps.ROIoffsetY = 0
	cameraProps.scaleFactor = 0.033

	xCoordinates = np.array([np.mgrid[0:imgSize],]*(imgSize)).transpose()
	yCoordinates = np.array([np.mgrid[0:imgSize],]*(imgSize))

	image = np.zeros((imgSize,imgSize))
	image[0][0] = 255

	#(height, center_x, center_y, width_x, width_y, n)
	centroids = []

	kinds = ['coif5']
	# kinds = pywt.wavelist(kind='discrete')
	# print(kinds)

	printTimeIncrement = 0.3 #s

	for kind in kinds:
		centroids = []
		for i in range(nbImages):
			if i <= 0 or i >= nbImages-1 or (time.perf_counter() - tWait > printTimeIncrement):
				print(f'Image {i+1}/{nbImages}')
				tWait = time.perf_counter()
			imgCentroid = 255*mm.gaussian(height+np.random.normal(0, heightSTD, size=1), centroid[0]+np.random.normal(0, centroidSTD, size=1), centroid[1]+np.random.normal(0, centroidSTD, size=1), sx+np.random.normal(0, sSTD, size=1), sy+np.random.normal(0, sSTD, size=1), n+np.random.normal(0, nSTD, size=1), np.random.randint(0,359))(xCoordinates,yCoordinates)
			totalImage = np.add(image, imgCentroid)
			noise = np.random.normal(noiseLevel, noiseSTD, size=(imgSize,imgSize))
			toCompute = np.add(totalImage, np.array(noise))
			toCompute = np.where(toCompute < 0, 0, toCompute)
			toCompute = np.where(toCompute > 255, 255, toCompute).astype(np.int_)
			
			# toCompute = denoise_tv_chambolle(toCompute, weight=0.1)
			# toCompute = denoise_tv_bregman(toCompute, weight=0.1) #:)
			# toCompute = denoise_bilateral(toCompute, sigma_spatial=2)
			# toCompute = denoise_nl_means(toCompute, sigma = 2)
			# toCompute = denoise_wavelet(toCompute, sigma=3, wavelet=kind, mode='soft', wavelet_levels=None, method='VisuShrink', rescale_sigma = True)
			
			centroids.append(compute_centroid(toCompute, cameraProps, i))

			# plt.figure()
			# plt.pcolormesh(xCoordinates, yCoordinates, toCompute, cmap=plt.cm.viridis)
			# plt.scatter(centroids[-1][2],centroids[-1][3],marker ='x',c='red')
			# plt.draw()
			# plt.pause(1e-17)
		
		x = [centroids[i][0] for i in range(nbImages)]
		y = [centroids[i][1] for i in range(nbImages)]

		xPx = [centroids[i][2] for i in range(nbImages)]
		yPx = [centroids[i][3] for i in range(nbImages)]

		print(f'{kind} : {np.mean(xPx):.6f}, {np.mean(yPx):.6f}, {np.std(x):.6f}, {np.std(y):.6f}, {np.max(x)-np.min(x):.6f}, {np.max(y)-np.min(y):.6f}')
		# plt.show()

	print(f'All done in {time.perf_counter()-tStart:.3f} [s]')

if __name__ == '__main__':
	_main()