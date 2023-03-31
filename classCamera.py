#cython: language_level=3

from pypylon import pylon as pypylon
from pypylon import genicam
import numpy as np
import sys
import os
from scipy import io
from skimage.filters import gaussian as gaussianFilter
import matplotlib.pyplot as plt
import miscmath as mm
import time
import DEFINES
import errors
from queue import Full
from computeCentroid import compute_centroid as compute_centroid

class CameraParameters:
	"""
	The class holding the camera parameters.

	Attributes
	----------
	cameraType: string
		The purpose of the camera. Either PC_CAMERA_TYPE_XY or PC_CAMERA_TYPE_TILT.
	maxX: uint
		The sensor pixel count in the X direction (width)
	maxY: uint
		The sensor pixel count in the Y direction (height)
	xCorr: np.ndarray
		The lens distortion correction in the X direction.
		There is 1 point per pixel, with the value being the X correction in pixels.
	yCorr: np.ndarray
		The lens distortion correction in the Y direction.
		There is 1 point per pixel, with the value being the Y correction in pixels.
	scaleFactor: float
		The scaling factor in mm/pixel
	ROICenter: tuple
		The center of the currently selected ROI
	validityRadius: float
		The validity radius. In case of a circular ROI, set this parameter to the desired ROI radius.
		Else, set this to PC_IMAGE_GET_ALL_ROI.\n
		See also miscmath.cropImage and miscmath.computeValidSoftROI
	ROIoffsetX: int
		The top left corner X coordinate of the ROI, in pixels
	ROIoffsetY: int
		The top left corner Y coordinate of the ROI, in pixels
	ROIwidth: uint
		The ROI width, in pixels
	ROIheight: uint
		The ROI height, in pixels
	minCropWindow: int
		The size in pixels of the smallest allowable crop.
		This value limits both the minimal crop height and the minimal crop width.
	nbImagesToGrab: int
		The number of images to grab to create one output image.
		Each output image is created by averaging over nbImagesToGrab to minimize the noise effects.
	ID: int
		The camera ID
	softROIrequired: bool
		When set to True, a software ROI will be computed in place of a hardware ROI.\n
		A hardware ROI is a ROI directly set on the camera and the received image is already cropped.\n
		A software ROI is a ROI that takes the whole image as an input and returns the cropped image after software calculations.

	Methods
	-------
	__init__:
		Initializes the class instances

	"""
	__slots__ = (	'cameraType',\
					'maxX',\
					'maxY',\
					'xCorr',\
					'yCorr',\
					'scaleFactor',\
					'ROICenter',\
					'validityRadius',\
					'ROIoffsetX',\
					'ROIoffsetY',\
					'ROIwidth',\
					'ROIheight',\
					'minCropWindow',\
					'nbImagesToGrab',\
					'ID',\
					'softROIrequired')

	def __init__(self, cameraType, camHandle= None):
		"""
		Initializes the class instances

		Parameters
		----------
		cameraType: string
			PC_CAMERA_TYPE_XY or PC_CAMERA_TYPE_TILT
		camHandle: pypylon.pylon.InstantCamera, optional
			The camera handle. This is used only to retrieve the maximal ROI of the camera and is not stored.

		"""
		self.cameraType = cameraType
		if not camHandle == None:
			self.maxX = camHandle.Width.Max
			self.maxY = camHandle.Height.Max
		else:
			self.maxX = 0
			self.maxY = 0

		self.xCorr = np.zeros((self.maxX, self.maxY))
		self.yCorr = np.zeros((self.maxX, self.maxY))
		
		self.ROICenter = (self.maxX/2,self.maxY/2)
		self.validityRadius = np.sqrt(self.maxX**2+self.maxY**2)
		self.ROIoffsetX = 0
		self.ROIoffsetY = 0
		self.ROIwidth = self.maxX
		self.ROIheight = self.maxY
		self.minCropWindow = max(self.maxX,self.maxY)
		self.nbImagesToGrab = 1

		self.ID = -1
		self.softROIrequired = False

class Camera:
	"""
	The camera class. It contains all the parameters and methods to interact with the Basler camera.

	Attributes
	----------
	connected: bool
		This is set to True when the camera is connected and is False otherwise
	parameters: classCamera.CameraParameters
		The camera parameters
	camHandle: pypylon.pylon.InstantCamera
		The camera handle.

	Methods
	-------
	__init__:
		Initializes the class instances
	__del__:
		Correctly disconnects the camera before deleting itself
	connect:
		Connects to a camera. If an other camera is already connected, it will be disconnected first.
	setDistortionCorrection:
		Opens a camera distortion correction file and loads its data in the camera parameters
	setROI:
		Changes the current ROI of the camera.
	setMaxROI:
		Changes the current ROI of the camera to its maximal value, i.e. to the whole field of view.
	getROI:
		Returns the current ROI as stored in the camera parameters.
	computeValidSoftROI:
		Computes the validity region and sets any pixel outside to 0.
	getImage:
		Requests the camera to grab an image.
	getBurstImages:
		Requests the camera to grab a burst of images
	getOptimalExposure:
		Sets the optimal exposure for the camera and returns it.
	setProperties:
		Change the camera configuration
	setExposure:
		Change the camera exposure
	close:
		Disconnects the camera
	getAvailableCameraIDs:
		Returns the available camera IDs

	"""

	__slots__ = (	'connected',\
					'parameters',\
					'camHandle')
	
	def __init__(self, cameraType = None, compatibleCameraID = None):
		"""
		Initializes the class instances

		Parameters
		----------
		cameraType: string, optional
			PC_CAMERA_TYPE_XY or PC_CAMERA_TYPE_TILT
		compatibleCameraID: int, optional
			The camera ID to connect to.
			If None or left blanck, it will connect to the first camera detected.

		"""

		self.connected = False
		self.parameters = CameraParameters(cameraType)
		self.camHandle = None
		if cameraType is not None:
			self.connect(cameraType, compatibleCameraID)
		
	def __del__(self):
		"""Correctly disconnects the camera before deleting itself"""

		if self.connected:
			try:
				self.camHandle.Close()
			except genicam.GenericException: 
				pass
			self.parameters = CameraParameters(None)

	def connect(self, cameraType, compatibleCameraID = None):
		"""
		Connects to a camera. If an other camera is already connected, it will be disconnected first.

		Parameters
		----------
		cameraType: string
			PC_CAMERA_TYPE_XY or PC_CAMERA_TYPE_TILT
		compatibleCameraID: int, optional
			The camera ID to connect to.
			If None or left blanck, it will connect to the first camera detected.
	
		Raises
		------
		errors.CameraError:
			If the camera connection failed.

		"""

		if not self.connected or not (self.parameters.cameraType == cameraType) or not (self.parameters.ID == compatibleCameraID) :
			#If we want to change the camera, close the currently connected one
			if self.connected:
				try:
					self.camHandle.close()
				except genicam.GenericException:
					pass
			#Connect to the new camera		
			try:
				if cameraType == None:
					raise errors.CameraError("No camera type specified at initialization")

				#initalize the camera
				tlf = pypylon.TlFactory.GetInstance()

				available_cameras = tlf.EnumerateDevices()
				available_ids = np.zeros(len(available_cameras))

				for i in range(0,len(available_cameras)):
					available_ids[i] = available_cameras[i].GetSerialNumber()
					if available_ids[i] == compatibleCameraID or compatibleCameraID == None:
						cameraAlreadyUsed = False
						try:
							self.camHandle = pypylon.InstantCamera(tlf.CreateDevice(available_cameras[i]))
						except genicam.GenericException:
							cameraAlreadyUsed = True
							if not compatibleCameraID == None:
								raise errors.CameraError("No compatible camera could be found") from None #If this was a specific ID search, break

						if not cameraAlreadyUsed:
							self.camHandle.Open()

							#configure default camera settings
							self.camHandle.PixelFormat 	= DEFINES.PC_CAMERA_PIXEL_FORMAT
							self.camHandle.OffsetX 		= 0
							self.camHandle.OffsetY 		= 0
							self.camHandle.Width 		= self.camHandle.Width.Max
							self.camHandle.Height 		= self.camHandle.Height.Max

							self.parameters.cameraType = cameraType
							self.parameters.maxX = self.camHandle.Width.Max
							self.parameters.maxY = self.camHandle.Height.Max
							self.parameters.ID = int(available_ids[i])

							if cameraType == DEFINES.PC_CAMERA_TYPE_XY:
								self.parameters.nbImagesToGrab		= DEFINES.PC_CAMERA_XY_NB_IMAGES_PER_POINT
								self.parameters.minCropWindow		= DEFINES.PC_CAMERA_XY_MIN_CROP_WINDOW
							else:
								self.parameters.nbImagesToGrab		= DEFINES.PC_CAMERA_TILT_NB_IMAGES_PER_POINT
								self.parameters.minCropWindow		= DEFINES.PC_CAMERA_FORBID_CROPPING

							if self.parameters.minCropWindow == DEFINES.PC_CAMERA_FORBID_CROPPING:
								self.parameters.minCropWindow = np.max(self.parameters.maxX, self.parameters.maxY)

							if self.camHandle.GetDeviceInfo().GetModelName() == 'acA5472-17um': #Require soft ROI for the acA5472-17um cameras
								self.parameters.softROIrequired = True

							self.connected = True
							return

				if not self.connected:
					raise errors.CameraError("No compatible camera could be found") from None #If this was a specific ID search, break

			except errors.CameraError as e:
				raise errors.CameraError("Camera initialization failed") from None
			except genicam.GenericException:
				raise errors.CameraError("Camera initialization failed") from None

	def setDistortionCorrection(self, config):
		"""
		Opens a camera distortion correction file and loads its data in the camera parameters

		Parameters
		----------
		config: classConfig.Config
			The program configuration state. This is used to retrieve the folders from which the file must be loaded.

		Raises
		------
		errors.IOError:
			If the camera distortion file could not be loaded properly.

		"""

		if self.connected:			 
			fileName = os.path.join(config.get_camera_path(), 'camera_'+str(self.parameters.ID)+config.cameraFileExtension)

			#load the camera distortion parameters
			try:
				cam_distortion = io.loadmat(fileName)
			except IOError:
				raise errors.IOError("Camera distortion file could not be loaded") from None

			try:
				cam_distortion = cam_distortion[DEFINES.PC_FILE_DISTORTION_PARAMETERS_NAME]

				cam_x_corr = cam_distortion[DEFINES.PC_FILE_DISTORTION_XCORR_NAME]
				cam_y_corr = cam_distortion[DEFINES.PC_FILE_DISTORTION_YCORR_NAME]
				cam_scale_factor = cam_distortion[DEFINES.PC_FILE_DISTORTION_SCALE_FACTOR_NAME]

				self.parameters.xCorr = np.nan_to_num(cam_x_corr[0,0])
				self.parameters.yCorr = np.nan_to_num(cam_y_corr[0,0])
				self.parameters.scaleFactor = cam_scale_factor[0,0][0,0]
			except IOError:
				raise errors.IOError("Camera distortion file data is corrupted") from None

	def setROI(self, ROI):
		"""
		Changes the current ROI of the camera.

		Parameters
		----------
		ROI: tuple
			The Region Of Interest in the following form:\n
			1- (int) X coordinate of the center of the ROI\n
			2- (int) Y coordinate of the center of the ROI\n
			3- (int) Width (X-span) of the ROI\n
			4- (int) Height (Y-span) of the ROI\n
			5- (int) The validity radius from the center. Set this to PC_IMAGE_GET_ALL_ROI to avoid any circular crop.

		Raises
		------
		errors.OutOfRangeError:
			If the ROI is off-limits
		errors.CameraError:
			If the camera did not accept the ROI or got disconnected.

		"""

		if self.connected:
			if ROI[0] > self.parameters.maxX or ROI[0] < 0 or ROI[1] > self.parameters.maxY or ROI[1] < 0:
				raise errors.OutOfRangeError("Specified camera ROI is out of range")

			try:
				x_min = int(ROI[0]-ROI[2]/2)
				x_max = int(ROI[0]+ROI[2]/2)
				y_min = int(ROI[1]-ROI[3]/2)
				y_max = int(ROI[1]+ROI[3]/2)

				#crop ROI while it is not in the image, up to the minimal window allowed
				if x_min < 0:
					x_min = 0
				if x_max-x_min < self.parameters.minCropWindow:
					x_max = x_min+self.parameters.minCropWindow
				if x_max > self.parameters.maxX:
					x_max = self.parameters.maxX
				if x_max-x_min < self.parameters.minCropWindow:
					x_min = x_max-self.parameters.minCropWindow
				if y_min < 0:
					y_min = 0
				if y_max-y_min < self.parameters.minCropWindow:
					y_max = y_min+self.parameters.minCropWindow
				if y_max > self.parameters.maxY:
					y_max = self.parameters.maxY
				if y_max-y_min < self.parameters.minCropWindow:
					y_min = y_max-self.parameters.minCropWindow

				if self.camHandle.GetDeviceInfo().GetModelName() == 'acA5472-17um': #Strangly, the acA5472-17um camera needs a multiple of 4 for Xoffset and width
					x_min = x_min-x_min%4
					width = x_max-x_min
					width = width - width%4
				else:
					width = x_max-x_min

				height = y_max-y_min

				self.parameters.ROICenter = (ROI[0],ROI[1])
				self.parameters.validityRadius = ROI[4]
				self.parameters.ROIoffsetX = x_min
				self.parameters.ROIoffsetY = y_min
				self.parameters.ROIwidth = width
				self.parameters.ROIheight = height
				
				#set the ROI properties
				if  self.camHandle.OffsetX.Value + width < self.parameters.maxX:
					self.camHandle.Width  = width
					self.camHandle.OffsetX= x_min
				else:
					self.camHandle.OffsetX= x_min
					self.camHandle.Width  = width
		
				if  self.camHandle.OffsetY.Value + height < self.parameters.maxY:
					self.camHandle.Height = height
					self.camHandle.OffsetY= y_min
				else:
					self.camHandle.OffsetY= y_min
					self.camHandle.Height = height
			except genicam.GenericException:
				self.connected = False
				raise errors.CameraError("Camera communication failed")

	def setMaxROI(self):
		"""
		Changes the current ROI of the camera to its maximal value, i.e. to the whole field of view.

		Raises
		------
		errors.CameraError:
			If the camera did not accept the ROI or got disconnected.

		"""
		if self.connected:
			try:
				self.parameters.ROICenter = (self.parameters.maxX,self.parameters.maxY)
				self.parameters.validityRadius = DEFINES.PC_IMAGE_GET_ALL_ROI
				self.parameters.ROIoffsetX = 0
				self.parameters.ROIoffsetY = 0
				self.parameters.ROIwidth = self.parameters.maxX
				self.parameters.ROIheight = self.parameters.maxY
				self.camHandle.OffsetX= 0
				self.camHandle.Width  = self.camHandle.Width.Max
				self.camHandle.OffsetY= 0
				self.camHandle.Height = self.camHandle.Height.Max
			except genicam.GenericException:
				self.connected = False
				raise errors.CameraError("Camera communication failed")

	def getROI(self):
		"""
		Returns the current ROI as stored in the camera parameters.
		
		Returns
		-------
		Tuple with the currently set ROI:
			1- (int) X coordinate of the top left corner\n
			2- (int) Y coordinate of the top left corner\n
			3- (int) Width (X-span) of the ROI\n
			4- (int) Height (Y-span) of the ROI

		Raises
		------
		errors.CameraError:
			If the camera is disconnected.

		"""

		if not self.connected:
			raise errors.CameraError("Camera is not connected") from None
		else:
			return self.parameters.ROIoffsetX,self.parameters.ROIoffsetY,self.parameters.ROIwidth,self.parameters.ROIheight
		
	def computeValidSoftROI(self, image, validityCenter, validityRadius):
		"""
		Computes the validity region and sets any pixel outside to 0.
		
		Parameters
		----------
		image: np.ndarray
			The image to perform the crop onto. This has to be a 2-dimensionnal array containing the pixel value, in int or float.
		validityCenter: tuple
			The coordinates of the center of the validity region.\n
			1- The X coordinate\n
			2- The Y coordinate
		validityRadius: float
			The radius of the validity region. If set to PC_IMAGE_GET_ALL_ROI, the function does nothing

		Returns
		-------
		Tuple: image, offsetX, offsetY
			1- (np.ndarray) The cropped image. Any pixel outside the validityRadius is set to 0. The input image size is reduces such as there is no zero-only column nor line.\n
			2- (int) The X offset. If the image was reduced, this is the distance in pixels from the old leften edge to the new one.\n
			3- (int) The Y offset  If the image was reduced, this is the distance in pixels from the old top edge to the new one.

		"""

		if validityRadius == DEFINES.PC_IMAGE_GET_ALL_ROI:
			return image, 0, 0
		else:
			return mm.computeValidSoftROI(image, self.parameters.maxX, self.parameters.maxY, validityCenter, validityRadius)

	def getImage(self, centroidQueue = None, imageID = None):
		"""
		Requests the camera to grab an image.

		The returned image is the average of nbImagesToGrab numbers of image, as specified in the camera.parameters.\n
		If the centroidQueue and imageID are provided, the image is immediately sent to the centroid computation queue.
		
		Parameters
		----------
		centroidQueue (optional): multiprocessing.Queue
			The centroid computation queue. If provided, the image will be immediately sent to the centroid computation processes.
		imageID: int, optional
			The image unique identifier used to track when it was taken.\n 
			See miscmath.generate_img_ID for more details.

		Returns
		-------
		picture: np.ndarray
			The averaged image. 

		Raises
		------
		errors.CameraError:
			- If the camera was not able to grab the image or got disconnected.
		errors.Error:
			- If the program could not take an image because of insufficient RAM more than MAX_MEMORY_RECOVERY_TRIES times in a row\n
			- If the program could not place an item in the queue because of insufficient RAM more than MAX_QUEUE_RECOVERY_TRIES times in a row

		"""


		if not self.connected:
			raise errors.CameraError("Camera is not connected") from None
		else:
			for watchdog in range(DEFINES.MAX_MEMORY_RECOVERY_TRIES):
				try:
					#Initialize result container
					i = 0
					result = np.zeros((self.camHandle.Height.Value, self.camHandle.Width.Value), dtype = np.uint32)

					#grab images			
					images = self.getBurstImages(self.parameters.nbImagesToGrab, computeValidity = False)
					for image in images:
						result = np.add(image, result)
					result = np.divide(result,len(images))
					

					#crop validity circle
					validityCenter = (self.parameters.ROICenter[0]-self.camHandle.OffsetX.Value,self.parameters.ROICenter[1]-self.camHandle.OffsetY.Value)
					if self.parameters.validityRadius != DEFINES.PC_IMAGE_GET_ALL_ROI:
						circularMask = mm.create_circular_mask(result.shape[0], result.shape[1], validityCenter, self.parameters.validityRadius)
						
						result[~circularMask] = 0

					picture = result.astype(np.uint16)
					# t7 = time.perf_counter()
					# print(f'NP Divide image\t\t\t{t7-t6:3.3f}')
					#directly send to computation queue if asked for
					if centroidQueue is not None:
						for watchdog2 in range(DEFINES.MAX_QUEUE_RECOVERY_TRIES):
							try:
								centroidQueue.put((picture, self.parameters.ROIoffsetX, self.parameters.ROIoffsetY, imageID, validityCenter, self.parameters.validityRadius), block = True, timeout = DEFINES.PROCESS_QUEUE_RECOVERY_DELAY)
								break
							except (MemoryError, Full):
								if watchdog2 >= DEFINES.MAX_QUEUE_RECOVERY_TRIES-1:
									raise errors.Error('Computation queue is out of memory and could not be recovered.')
								time.sleep(DEFINES.PROCESS_QUEUE_RECOVERY_DELAY)

					# t8 = time.perf_counter()
					# print(f'Total execution\t\t\t{t8-t0:3.3f}')
					#return image
					return picture

				except (genicam.GenericException, SystemError):
					self.connected = False
					raise errors.CameraError("Camera communication failed during image grabbing")
				except MemoryError:
					if watchdog >= DEFINES.MAX_MEMORY_RECOVERY_TRIES-1:
						raise errors.Error('Program is out of memory and could not be recovered.')
					time.sleep(DEFINES.PROCESS_MEMORY_RECOVERY_DELAY)

	def getBurstImages(self, nbImages, computeValidity = True):
		"""
		Requests the camera to grab a burst of images

		Parameters
		----------
		nbImages: int
			The number of images to grab.
		computeValidity: bool, optional
			If False, the images will not be cropped to the required ROI stored in the parameters.\n
			If True or blanck, it will be cropped.

		Returns
		-------
		images: list of np.ndarray
			The grabbed images. 

		Raises
		------
		errors.CameraError:
			If the camera was not able to grab the images or got disconnected.\n
			If the program is out of memory

		"""

		if not self.connected:
			raise errors.CameraError("Camera is not connected") from None
		else:
			try:
				self.camHandle.StartGrabbingMax(nbImages)
				images = []
				while self.camHandle.IsGrabbing():
					grabResult = self.camHandle.RetrieveResult(DEFINES.PC_IMAGE_TIMEOUT, pypylon.TimeoutHandling_Return)
					
					if grabResult.GrabSucceeded():
						images.append((grabResult.Array).astype(np.uint16))
						grabResult.Release()

				#crop validity circle
				if computeValidity:
					validityCenter = (self.parameters.ROICenter[0]-self.camHandle.OffsetX.Value,self.parameters.ROICenter[1]-self.camHandle.OffsetY.Value)
					if self.parameters.validityRadius != DEFINES.PC_IMAGE_GET_ALL_ROI:
						for image in images:
							circularMask = mm.create_circular_mask(image.shape[0], image.shape[1], validityCenter, self.parameters.validityRadius)
							
							image[~circularMask] = 0

				return images

			except (genicam.GenericException, SystemError):
				self.connected = False
				raise errors.CameraError("Camera communication failed during image burst grabbing")
			
	def getOptimalExposure(self, initExposure):
		"""
		Sets the optimal exposure for the camera and returns it.
		
		Parameters
		----------
		initExposure: float
			The initial exposure that the exposure optimization process will start from.
		
		Returns
		-------
		currentExposure: float
			The optimal exposure as determined by the optimization process

		Raises
		------
		errors.CameraError:
			If the camera is not connected.\n
			If the camera was not able to set the exposure or got disconnected.

		"""

		#Grab an image, filter it, check if the maximal value if OK and repeat if not. If it is OK, return the value.

		if not self.connected:
			raise errors.CameraError("Camera is not connected")
		else:
			try:
				#init loop
				i = 1
				nbOk = 0
				if self.parameters.cameraType == DEFINES.PC_CAMERA_TYPE_XY:
					maxExposure = DEFINES.PC_CAMERA_XY_MAX_EXPOSURE
				else:
					maxExposure = DEFINES.PC_CAMERA_GET_EXPOSURE_EXPOSURE_MAX

				currentExposure = initExposure
				
				self.camHandle.ExposureTime = currentExposure
				
				#grab images and adapt exposure
				while i <= DEFINES.PC_CAMERA_GET_EXPOSURE_MAX_ITERATIONS:
					#get one image
					
					image = self.camHandle.GrabOne(DEFINES.PC_IMAGE_TIMEOUT)
					image = np.divide(image.Array,DEFINES.PC_CAMERA_MAX_INTENSITY_RAW)

					#crop validity circle
					
					validityCenter = (self.parameters.ROICenter[0]-self.camHandle.OffsetX.Value,self.parameters.ROICenter[1]-self.camHandle.OffsetY.Value)
					image, offsetX, offsetY = self.computeValidSoftROI(image, validityCenter, self.parameters.validityRadius)

					#filter the image with a gaussian filter
					if self.parameters.cameraType == DEFINES.PC_CAMERA_TYPE_XY:
						image = gaussianFilter(image,DEFINES.CC_IMAGE_XY_FILTERING_SIGMA)
					else:
						image = gaussianFilter(image,DEFINES.CC_IMAGE_TILT_FILTERING_SIGMA)

					maxPxIntensity = np.max(image)

					#Check the stopping criterias
					if (maxPxIntensity >= DEFINES.PC_CAMERA_GET_EXPOSURE_TARGET_INTENSITY-DEFINES.PC_CAMERA_GET_EXPOSURE_INTENSITY_TOLERANCE and \
						maxPxIntensity <= DEFINES.PC_CAMERA_GET_EXPOSURE_TARGET_INTENSITY+DEFINES.PC_CAMERA_GET_EXPOSURE_INTENSITY_TOLERANCE) or \
						currentExposure == DEFINES.PC_CAMERA_GET_EXPOSURE_EXPOSURE_MIN or \
						currentExposure == maxExposure:
						nbOk += 1

					#Check if the stopping condition is reached
					if nbOk >= DEFINES.PC_CAMERA_GET_EXPOSURE_NB_OK:
						return currentExposure

					#Adapt the exposure
					if maxPxIntensity >= DEFINES.PC_CAMERA_GET_EXPOSURE_SATURED_THRESHOLD:
						currentExposure = currentExposure * DEFINES.PC_CAMERA_GET_EXPOSURE_TARGET_INTENSITY/maxPxIntensity*DEFINES.PC_CAMERA_GET_EXPOSURE_SATURED_GAIN #if we are satured, decrease exposure faster
					else:
						currentExposure = currentExposure * DEFINES.PC_CAMERA_GET_EXPOSURE_TARGET_INTENSITY/maxPxIntensity

					if currentExposure < DEFINES.PC_CAMERA_GET_EXPOSURE_EXPOSURE_MIN:
						currentExposure = DEFINES.PC_CAMERA_GET_EXPOSURE_EXPOSURE_MIN
					elif currentExposure > maxExposure:
						currentExposure = maxExposure

					self.camHandle.ExposureTime = currentExposure
					i = i+1


				return currentExposure
			except genicam.GenericException:
				self.connected = False
				raise errors.CameraError("Camera communication failed during optimal exposure determination")
		
	def setProperties(self, exposure, gain, blackLevel, gamma):
		"""
		Change the camera configuration
		
		Parameters
		----------
		exposure: float
			The new exposure
		gain: float
			The new camera gain
		blackLevel: float
			The new camera black level
		gamma: float
			The new camera gamma value

		Raises
		------
		errors.CameraError:
			If the camera was not able to set the parameters or got disconnected.

		"""

		if self.connected:
			#modify properties
			try:
				self.camHandle.ExposureTime 	= exposure
				self.camHandle.Gain 			= gain
				self.camHandle.BlackLevel 		= blackLevel
				self.camHandle.Gamma 			= gamma
			except genicam.GenericException:
				self.connected = False
				raise errors.CameraError("Camera communication failed")
			
	def setExposure(self, exposure):
		"""
		Change the camera exposure
		
		Parameters
		----------
		exposure: float
			The new exposure

		Raises
		------
		errors.CameraError:
			If the camera was not able to set the exposure or got disconnected.

		"""

		if self.connected:
			try:
				self.camHandle.ExposureTime 	= exposure
			except genicam.GenericException:
				self.connected = False
				raise errors.CameraError("Camera communication failed")
			
	def close(self):
		"""Disconnects the camera"""

		if self.connected:
			try:
				self.camHandle.Close()
			except genicam.GenericException:
				pass
		self.connected = False

	def getAvailableCameraIDs(self):
		"""Returns the available camera IDs"""

		return getAvailableCameraIDs()

	def getCentroid(self):
                """Return the centroid found on the image"""
                self.setMaxROI()
                picture = self.getImage()
                centroid = compute_centroid(picture, self.parameters, 1)
                return centroid[0], centroid[1]
                
                
                
                

def getAvailableCameraIDs():
	"""
	Returns the available camera IDs

	Returns
	-------
	available_ids: list
		The list of camera serial numbers connected to the computer

	Raises
	------
	errors.CameraError:
		If the communication with a camera failed

	"""

	try:
		available_cameras = pypylon.TlFactory.GetInstance().EnumerateDevices()
		available_ids = []

		for i in range(0,len(available_cameras)):
			available_ids.append(int(available_cameras[i].GetSerialNumber()))
		return available_ids
	except genicam.GenericException:
		raise errors.CameraError("Camera communication failed")

def _graph_heating_effect():
	import matplotlib.pyplot as plt
	from computeCentroid import compute_centroid as compute_centroid
	import classGeneral

	centroids_preheat_path = os.path.join('Python_garbage','centroids_preheat_vent_0.mat')
	loadFromFile = False
	image_ID = 0
	pollPeriod = 0.3 #second between two images. MIN is 0.3
	RMSFilterTimeWindow = 5 #seconds used for the RMS filter window
	preheatTime = 20*60 #seconds
	cooldownTime = 0*60 #seconds
	totalPts = preheatTime/pollPeriod

	allCentroids = np.full((int(totalPts),8),np.nan)
	allTimes = np.full((int(totalPts)),np.nan)
	RMSFiltered = np.full((int(totalPts),3),np.nan)

	if loadFromFile:
		previousRun = io.loadmat(centroids_preheat_path)
		allCentroids = previousRun['centroids']
		allTimes = previousRun['time'][0]
		RMSFiltered = previousRun['RMSFiltered']

	if not loadFromFile:
		general = classGeneral.General()
		general.config.currentTestBenchFile = '05_XY_7bench_2'
		general.testBench.load(general.config.get_current_testBench_fileName())
		general.testBench.init_handles(general.config)
		general.genericPositioner.model.clear(general.genericPositioner.physics)
		general.testBench.search_positioners(general.genericPositioner)

		general.testBench.cameraXY.setMaxROI()
		completeExposure = general.testBench.cameraXY.getOptimalExposure(DEFINES.PC_CAMERA_XY_DEFAULT_EXPOSURE)

		centroid = compute_centroid(general.testBench.cameraXY.getImage(), general.testBench.cameraXY.parameters, image_ID)
		ROI = np.zeros(5, dtype = np.uint16)
		ROI[0] = int(centroid[2])
		ROI[1] = int(centroid[3])
		ROI[2] = general.testBench.cameraXY.parameters.minCropWindow+20
		ROI[3] = general.testBench.cameraXY.parameters.minCropWindow
		ROI[4] = np.sqrt(ROI[2]**2+ROI[3]**2)
		general.testBench.cameraXY.setROI(ROI)

		for positioner in general.testBench.positioners:
			positioner.change_open_loop_status(general.testBench.canUSB, enable = False)
			positioner.set_current(general.testBench.canUSB, positioner.physics.maxCurrentAlpha, positioner.physics.maxCurrentBeta)

		fig = plt.figure()
		ax = plt.subplot(1,1,1)

		tStart = time.time()
		#Wait until the bench is sufficiently hot
		currentPt = 0
		i = 0
		while tStart + preheatTime > time.time():
			tSync = time.time()

			allCentroids[currentPt, :] = (compute_centroid(general.testBench.cameraXY.getImage(), general.testBench.cameraXY.parameters, image_ID))
			allTimes[currentPt] = (time.time()-tStart)

			if currentPt >= RMSFilterTimeWindow / pollPeriod:
				mean1 = np.nanmean(allCentroids[i:currentPt+1,0])
				mean2 = np.nanmean(allCentroids[i:currentPt+1,1])
				err1 = allCentroids[i:currentPt+1,0] - mean1
				err2 = allCentroids[i:currentPt+1,1] - mean2
				RMSFiltered[i,:] = (1000*mm.nanrms(err1),1000*mm.nanrms(err2),allTimes[currentPt])
				i+= 1
				ax.clear()
				ax.plot(RMSFiltered[max(0,i-int(RMSFilterTimeWindow / pollPeriod)):,2],RMSFiltered[max(0,i-int(RMSFilterTimeWindow / pollPeriod)):,0], color = 'orange')
				ax.set_ylim(0,1)
				plt.draw()
				plt.pause(1e-17)

			currentPt += 1

			(days, hours, minutes, seconds) = mm.decompose_time(tStart + preheatTime- time.time())
			while tSync+pollPeriod>time.time():
				_=0

		general.testBench.set_current_all_positioners(0,0)

		# while tStart + preheatTime + cooldownTime > time.time():
		# 	tSync = time.time()
		# 	allCentroids.append(compute_centroid(general.testBench.cameraXY.getImage(), general.testBench.cameraXY.parameters, image_ID))
		# 	allTimes.append(time.time()-tStart)
		# 	(days, hours, minutes, seconds) = mm.decompose_time(tStart + preheatTime + cooldownTime- time.time())
		# 	while tSync+pollPeriod>time.time():
		# 		_=0

	allCentroids = np.asarray(allCentroids)
	allTimes = np.asarray(allTimes)

	if len(allCentroids) >= RMSFilterTimeWindow / pollPeriod:
		for i in range(0,int(len(allCentroids)-RMSFilterTimeWindow / pollPeriod)):
			mean1 = np.nanmean(allCentroids[i:int(i+RMSFilterTimeWindow / pollPeriod),0])
			mean2 = np.nanmean(allCentroids[i:int(i+RMSFilterTimeWindow / pollPeriod),1])
			err1 = allCentroids[i:int(i+RMSFilterTimeWindow / pollPeriod),0] - mean1
			err2 = allCentroids[i:int(i+RMSFilterTimeWindow / pollPeriod),1] - mean2
			RMSFiltered.append([1000*mm.nanrms(err1),1000*mm.nanrms(err2),allTimes[int(i+RMSFilterTimeWindow / pollPeriod)]])

	RMSFiltered = np.asarray(RMSFiltered)

	if not loadFromFile:
		io.savemat(centroids_preheat_path, \
				{	'centroids': allCentroids,
					'time': allTimes,
					'RMSFiltered': RMSFiltered})

		general.stop_all()


	plt.figure()
	plt.plot(allTimes, allCentroids[:,0], color = 'red')
	plt.figure()
	plt.plot(allTimes, allCentroids[:,1], color = 'green')
	plt.figure()
	plt.plot(allTimes, np.sqrt(allCentroids[:,2]**2+allCentroids[:,3]**2), color = 'darkblue')
	plt.figure()
	plt.plot(RMSFiltered[:,2], RMSFiltered[:,0], color = 'orange')
	plt.figure()
	plt.plot(RMSFiltered[:,2], RMSFiltered[:,1], color = 'orange')

	plt.show()

def _main(nbCentroidsLoop = 1000):
	import matplotlib.image as mpimg
	import matplotlib.pyplot as plt
	from computeCentroid import compute_centroid as compute_centroid
	import classConfig
	import numpy as np


	# try:
	# 	try:
	# 		raise OSError
	# 	except OSError:
	# 		raise errors.CameraError("error1")

	# 	try:
	# 		raise OSError
	# 	except OSError:
	# 		raise errors.CameraError("error2")
	# except Exception as e:
	# 	print(e)

	#Set the default values
	# im_path = os.path.join('Python_garbage','garbage.png')
	im_path = ''

	nbComputationsPerCentroid = 1
	nbImages = [1]
	centroids_txt_path = os.path.join('Python_garbage','centroids.mat')
	os.makedirs('Python_garbage', exist_ok=True)

	image_ID = 1
	exposure_time = 1000
	gain = 0.0
	black_level = 1.25
	gamma = 1.0

	ROI = np.zeros(5, dtype = np.uint16)

	#python grab_image.py "" "..\41000-Matlab-Calibration_and_test\calibration_data_cam1_XY.mat" 1 12345 4 11111 2 2.5 1.5 1.5
	nb_args = len(sys.argv)-1

	if nb_args > 0:
		im_path = sys.argv[1]
	if nb_args > 1:
		cam_path = sys.argv[2]
	if nb_args > 2:
		camera_ID = int(sys.argv[3])
	if nb_args > 3:
		image_ID = int(sys.argv[4])
	if nb_args > 4:
		nb_images = int(sys.argv[5])
	if nb_args > 5:
		exposure_time = float(sys.argv[6])
	if nb_args > 6:
		gain = float(sys.argv[7])
	if nb_args > 7:
		black_level = float(sys.argv[8])
	if nb_args > 8:
		gamma = float(sys.argv[9])

	if im_path != '' or image_ID > -1:
		centroid = (0,0,0,0,0,0,0,0)

		try:
			#create camera instance
			camera = Camera(cameraType = DEFINES.PC_CAMERA_TYPE_XY)#, compatibleCameraID=22994237) #22942361 #22994237

			#_=os.system('echo Taking picture')
			camera.setMaxROI()
			exposure_time = camera.getOptimalExposure(exposure_time)
			if exposure_time >= DEFINES.PC_CAMERA_XY_MAX_EXPOSURE:
				camera.close()
				return
			camera.setProperties(exposure_time, gain, black_level, gamma)

			if image_ID > -1:
				config = classConfig.Config()
				camera.setDistortionCorrection(config)

			centroids = np.zeros((nbCentroidsLoop,3))
			minimum = 1000000*np.ones(2)
			maximum = 0*np.ones(2)
			mean = 0*np.ones(2)
			std = 0*np.ones(2)

			#Get first centroid to set ROI
			picture = camera.getImage()

			#Save the image
			if im_path != '':
				mpimg.imsave(im_path,picture)

			#compute the centroid
			if image_ID > -1:
				centroid = compute_centroid(picture, camera.parameters, image_ID)

			if np.isnan(centroid[0]):
				return

			ROI[0] = int(centroid[2])
			ROI[1] = int(centroid[3])
			ROI[2] = camera.parameters.minCropWindow+20
			ROI[3] = camera.parameters.minCropWindow
			ROI[4] = np.sqrt(ROI[2]**2+ROI[3]**2)

			camera.setROI(ROI)

			results = []
			for nbImage in nbImages:
				centroids = np.zeros((nbCentroidsLoop,3))
				minimum = 1000000*np.ones(2)
				maximum = 0*np.ones(2)
				mean = 0*np.ones(2)
				std = 0*np.ones(2)
				camera.parameters.nbImagesToGrab = nbImage
				for i in range(0,nbCentroidsLoop):
					currentComputation = []
					if nbComputationsPerCentroid>1:
						images = camera.getBurstImages(nbComputationsPerCentroid)
					else:
						images = [camera.getImage()]

					for image in images:
						t0 = time.perf_counter()

						#Save the image
						if im_path != '':
							mpimg.imsave(im_path,image)

						#compute the centroid
						t1 = time.perf_counter()
						if image_ID > -1:
							currentComputation.append(compute_centroid(image, camera.parameters, image_ID))
							

						if np.isnan(centroid[0]):
							raise errors.CameraError("Fiber light went out")

					currentComputation = np.array(currentComputation)
					# plotColors = plt.cm.gist_rainbow(np.linspace(0,1,len(currentComputation)))
					# plt.figure()
					# plt.scatter(currentComputation[:,0], currentComputation[:,1], color = plotColors)
					# plt.draw()
					# plt.pause(1e-10)

					# print(currentComputation)
					# print(currentComputation[0])
					# print(currentComputation[0,0])

					centroid = (1000*np.mean(currentComputation[:,0]),\
								1000*np.mean(currentComputation[:,1]),\
								1000*np.mean(currentComputation[:,2]),\
								1000*np.mean(currentComputation[:,3]),\
								1000*np.mean(currentComputation[:,4]),\
								1000*np.mean(currentComputation[:,5]),\
								np.mean(currentComputation[:,6]),\
								currentComputation[0,7])

					centroids[i,2] = i
					for j in range(0,2):
						centroids[i,j] = centroid[j]
						if centroid[j]<minimum[j]:
							minimum[j] = centroid[j]
						if centroid[j]>maximum[j]:
							maximum[j] = centroid[j]
						mean[j] = np.mean(centroids[0:(i+1),j])
						std[j] = np.std(centroids[0:(i+1),j])

					
				centroids[:,0] = centroids[:,0]-np.nanmean(centroids[:,0])
				centroids[:,1] = centroids[:,1]-np.nanmean(centroids[:,1])
				results.append(np.array([nbImage, mean[0],mean[1],std[0],std[1], maximum[0]-minimum[0], maximum[1]-minimum[1]]))

				plt.figure()
				plt.title(f'Deviation. nbImages = {nbImage}, STD: {np.sqrt(std[0]**2+std[1]**2):.3f} [um]')
				plt.scatter(centroids[:,0], centroids[:,1], color = 'red',marker = 'x', s = 1)
				plt.draw()
				plt.pause(1e-10)

			results = np.array(results)

		except errors.CameraError as e:
			return

		try:
			io.savemat(centroids_txt_path, \
				{'centroids': centroids})
		except OSError as e:
			return

		camera.close()

		# plt.figure()

		# plt.scatter(results[:,0], np.sqrt(results[:,3]**2+results[:,4]**2), color = 'red',marker = 'x', s = 1)
		# plt.scatter(results[:,0], np.sqrt(results[:,5]**2+results[:,6]**2), color = 'green',marker = 'x', s = 1)
		plt.show()



if __name__ == '__main__':
	#_graph_heating_effect()
	_main()
