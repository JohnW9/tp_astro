#cython: language_level=3
import time
from scipy import optimize,interpolate,stats
import numpy as np
import matplotlib.pyplot as plt
import copy
import DEFINES

MM_IMG_ID_BITSHIFT_FOR_CENTROID_TYPE		= 0
MM_IMG_ID_BITSHIFT_FOR_DIRECTION			= MM_IMG_ID_BITSHIFT_FOR_CENTROID_TYPE	+ DEFINES.MM_IMG_ID_BITS_FOR_CENTROID_TYPE
MM_IMG_ID_BITSHIFT_FOR_AXIS					= MM_IMG_ID_BITSHIFT_FOR_DIRECTION		+ DEFINES.MM_IMG_ID_BITS_FOR_DIRECTION
MM_IMG_ID_BITSHIFT_FOR_STEP					= MM_IMG_ID_BITSHIFT_FOR_AXIS			+ DEFINES.MM_IMG_ID_BITS_FOR_AXIS
MM_IMG_ID_BITSHIFT_FOR_REPETITION			= MM_IMG_ID_BITSHIFT_FOR_STEP			+ DEFINES.MM_IMG_ID_BITS_FOR_STEP
MM_IMG_ID_BITSHIFT_FOR_STARTING_POINT		= MM_IMG_ID_BITSHIFT_FOR_REPETITION		+ DEFINES.MM_IMG_ID_BITS_FOR_REPETITION
MM_IMG_ID_BITSHIFT_FOR_BENCH_SLOT			= MM_IMG_ID_BITSHIFT_FOR_STARTING_POINT	+ DEFINES.MM_IMG_ID_BITS_FOR_STARTING_POINT
MM_IMG_ID_BITMASK_FOR_CENTROID_TYPE			= (2**DEFINES.MM_IMG_ID_BITS_FOR_CENTROID_TYPE - 1) 	<< MM_IMG_ID_BITSHIFT_FOR_CENTROID_TYPE
MM_IMG_ID_BITMASK_FOR_DIRECTION				= (2**DEFINES.MM_IMG_ID_BITS_FOR_DIRECTION - 1) 		<< MM_IMG_ID_BITSHIFT_FOR_DIRECTION
MM_IMG_ID_BITMASK_FOR_AXIS					= (2**DEFINES.MM_IMG_ID_BITS_FOR_AXIS - 1) 				<< MM_IMG_ID_BITSHIFT_FOR_AXIS
MM_IMG_ID_BITMASK_FOR_STEP					= (2**DEFINES.MM_IMG_ID_BITS_FOR_STEP - 1) 				<< MM_IMG_ID_BITSHIFT_FOR_STEP
MM_IMG_ID_BITMASK_FOR_REPETITION			= (2**DEFINES.MM_IMG_ID_BITS_FOR_REPETITION - 1) 		<< MM_IMG_ID_BITSHIFT_FOR_REPETITION
MM_IMG_ID_BITMASK_FOR_STARTING_POINT		= (2**DEFINES.MM_IMG_ID_BITS_FOR_STARTING_POINT - 1) 	<< MM_IMG_ID_BITSHIFT_FOR_STARTING_POINT
MM_IMG_ID_BITMASK_FOR_BENCH_SLOT			= (2**DEFINES.MM_IMG_ID_BITS_FOR_BENCH_SLOT - 1) 		<< MM_IMG_ID_BITSHIFT_FOR_BENCH_SLOT

def convert_to_int_in_borns(item, minVal, maxVal, defaultVal):
	"""
	Converts an integer representation object to an integer and borns it between minVal and maxVal.

	Parameters
	----------
	item: string, float
		The object to be converted to an integer
	minVal: int
		The minimal value of the return value
	maxVal: int
		The maximal value of the return value
	defaultVal: int
		The value returned if the itam could not be converted

	Returns
	-------
	value: int
		The borned integer representation of the object. If it could not be converted, defaultVal is returned instead. 
	
	"""

	try:
		currentValue = int(item)
	except ValueError:
		currentValue = defaultVal

	if currentValue > maxVal:
		currentValue = maxVal
	elif currentValue < minVal:
		currentValue = minVal

	return currentValue

def convert_to_float_in_borns(item, minVal, maxVal, defaultVal, nbDecimals):
	"""
	Converts an float representation object to an float and borns it between minVal and maxVal.

	Parameters
	----------
	item: string, int
		The object to be converted to an float
	minVal: float
		The minimal value of the return value
	maxVal: float
		The maximal value of the return value
	defaultVal: float
		The value returned if the itam could not be converted

	Returns
	-------
	value: int
		The borned float representation of the object. If it could not be converted, defaultVal is returned instead. 
	
	"""

	try:
		currentValue = float(item)
	except ValueError:
		currentValue = defaultVal

	if currentValue > maxVal:
		currentValue = maxVal
	elif currentValue < minVal:
		currentValue = minVal

	return round(currentValue,nbDecimals)

def deg2rad(angle):
	"""Converts an angle in degrees to radians"""

	return angle*np.pi/180

def rad2deg(angle):
	"""Converts an angle in radians to degrees"""

	return angle*180/np.pi

def get_time_diff(startingTime):
	"""
	Returns a time difference using the time.time function

	Parameters
	----------
	startingTime: time.Time
		The starting time using the time module

	Returns
	-------
	dict of the decomposed time difference:
		'd': (int) The number of days\n
		'h': (int) The number of hours\n
		'm': (int) The number of minutes\n
		's': (float) The number of seconds, rounded to 1 decimal place
	
	"""
	t_current = time.time()

	ETA = (t_current-startingTime)
	
	ETA_d = int(round(ETA/86400, 0))
	ETA_h = int((ETA%86400)/3600)
	ETA_m = int((ETA%3600)/60)
	ETA_s = round((ETA%60),1)

	return {'d':ETA_d, 'h':ETA_h, 'm':ETA_m, 's':ETA_s}

def dist(p1,p2):
	"""
	Returns the distance between two points

	Parameters
	----------
	p1: list or tuple
		The x and y coordinates of the first point [x (float), y (float)]
	p2: list or tuple
		The x and y coordinates of the second point [x (float), y (float)]

	Returns
	-------
	float:
		The arithmetic distance between the two points

	"""

	return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def get_circumcenter(p1,p2,p3):
	"""
	Returns the circumcircle of a triangle.

	If the 3 points are not a valid triangle (colinear, coincident, etc.), the returned values will all be np.nan

	Parameters
	----------
	p1: list or tuple
		The x and y coordinates of the first point of the triangle [x (float), y (float)]
	p2: list or tuple
		The x and y coordinates of the second point of the triangle [x (float), y (float)]
	p3: list or tuple
		The x and y coordinates of the third point of the triangle [x (float), y (float)]

	Returns
	-------
	tuple: xc, yc, radius
		xc (float): x coordinate of the circumcenter\n
		yc (float): y coordinate of the circumcenter\n
		radius (float) the radius of the circumcircle

	"""

	ax = p1[0]
	ay = p1[1]
	bx = p2[0]
	by = p2[1]
	cx = p3[0]
	cy = p3[1]
	normA = np.sqrt(ax**2+ay**2)
	normB = np.sqrt(bx**2+by**2)
	normC = np.sqrt(cx**2+cy**2)
	xc = 0
	yc = 0
	Sx = 1/2*(normA**2*by+normB**2*cy+normC**2*ay-normA**2*cy-normB**2*ay-normC**2*by)
	Sy = 1/2*(normA**2*cx+normB**2*ax+normC**2*bx-normA**2*bx-normB**2*cx-normC**2*ax)
	normS = np.sqrt(Sx**2+Sy**2)
	a = (ax*by+bx*cy+cx*ay-ax*cy-bx*ay-cx*by)
	b = (ax*by*normC**2+bx*cy*normA**2+cx*ay*normB**2-ax*cy*normB**2-bx*ay*normC**2-cx*by*normA**2)

	if a == 0:
		return np.nan, np.nan, np.nan

	xc = Sx/a
	yc = Sy/a

	radius = np.sqrt(b/a+normS**2/a**2)

	return xc, yc, radius

def get_circle_center_approx(xData, yData):
	"""
	Returns an approximation for a center for the data provided. The data should represent a circle or an arc.

	It will cut the data list in 3 parts and use the median point of each section to construct a triangle.
	The circumcenter of the triangle is then computed and the coordinates are returned.

	Parameters
	----------
	xData: list of float
		The x coordinates of the points in the same order as yData
	yData: list of float
		The y coordinates of the points in the same order as xData

	Returns
	-------
	tuple: xC, yC
		xC (float): x coordinate of the center approximation\n
		yC (float): y coordinate of the center approximation

	"""

	circleSection = []
	nbData = len(xData)

	estimatePts = [int(0*nbData/3), int(1*nbData/3), int(2*nbData/3)]

	circleSection.append([xData[0:estimatePts[1]]				, yData[0:estimatePts[1]]])
	circleSection.append([xData[estimatePts[1]:estimatePts[2]]	, yData[estimatePts[1]:estimatePts[2]]])
	circleSection.append([xData[estimatePts[2]:-1]				, yData[estimatePts[2]:-1]])

	medianPoints = []
	medianPoints.append([np.nanmedian(circleSection[0][0]), np.nanmedian(circleSection[0][1])])
	medianPoints.append([np.nanmedian(circleSection[1][0]), np.nanmedian(circleSection[1][1])])
	medianPoints.append([np.nanmedian(circleSection[2][0]), np.nanmedian(circleSection[2][1])])

	(xC, yC, r) = get_circumcenter(	(medianPoints[0][0], medianPoints[0][1]),\
									(medianPoints[1][0], medianPoints[1][1]),\
									(medianPoints[2][0], medianPoints[2][1]))

	return (xC, yC)

def fit_circle(xData,yData):
	"""
	Fits a circle on the data points

	If there are 3 points or more in the data, it will first compute a rough approximation and then optimize it using least square minimization.
	If there are only 2 points in the data, then the center is set to the middle point and the radius to the half distance.
	If there are less points, the data is returned as is and the radius set to 0

	Parameters
	----------
	xData: list of float
		The x coordinates of the points in the same order as yData
	yData: list of float
		The y coordinates of the points in the same order as xData

	Returns
	-------
	tuple: centerX, centerY, radius
		centerX (float): the x coordinate of the center\n
		centerY (float): the y coordinate of the center\n
		radius (float): the radius of the circle

	"""

	#rough approximation of the circle's parameters

	xData = xData[~np.isnan(xData)]
	yData = yData[~np.isnan(yData)]

	nbData = len(xData)
	if nbData>2:
		#get rough estimation of circle using 3 well space points to create the circumcircle
		xData = xData.astype(np.float64)
		yData = yData.astype(np.float64)

		estimate_center = get_circle_center_approx(xData, yData)
		estimate_radius = np.nanmedian(dist((estimate_center[0],estimate_center[1]),(xData,yData)))

		params = (estimate_center[0], estimate_center[1], estimate_radius)

		#optimize
		errorfunction = lambda p: distToCircle(*p)(xData,yData)
		params, success = optimize.leastsq(errorfunction, params, ftol = 1e-30)
	elif nbData == 2:
		params = (np.mean(xData),np.mean(yData),np.sqrt((xData[0]-xData[1])**2+(yData[0]-yData[1])**2)/2)
	else:
		params = (xData,yData,0)

	return params

def intersect_circles(center1, r1, center2, r2):
	"""
	Returns the intersection points of 2 circles

	Parameters
	----------
	center1: tuple
		The center of the first circle in the form (x (float), y (float))
	r1: float
		The radius of the first circle
	center2: tuple
		The center of the second circle in the form (x (float), y (float))
	r2: float
		The radius of the second circle

	Returns
	-------
	list of intersection points
		intersection point: A list containing the x and y coordinates of the intersection [x (float), y (float)].

	"""

	cX1 = center1[0]
	cY1 = center1[1]
	cX2 = center2[0]
	cY2 = center2[1]
	dist = np.sqrt((cX1-cX2)**2+(cY1-cY2)**2)
	intersect = []

	if dist > r1+r2: 		# Circles are outside each other
		return intersect

	elif dist < abs(r2-r1): # One circle is fully in the other
		return intersect

	else:
		d = (r1**2)-(r2**2)-(cX1**2)+(cX2**2)-(cY1**2)+(cY2**2)
		e = 2*(cX1-cX2)
		f = 2*(cY1-cY2)

		if e != 0 and abs(e)>abs(f): #Equation solvable in x and with the best discriminant factor
			a = (f**2)/(e**2)+1
			b = (2*d*f)/(e**2)+(2*cX1*f)/(e)-(2*cY1)
			c = -((r1**2)-(d**2)/(e**2)-(cX1**2)-(cY1**2)-(2*cX1*d)/(e))

			delta = (b**2)-(4*a*c)
			if delta < 0:
				return intersect

			elif delta==0:
				x = -b/(2*a)
				y = (d+f*x)/(-e)
				intersect.append([x,y])
				return intersect

			else:
				y1 = (-b+np.sqrt(delta))/(2*a)
				y2 = (-b-np.sqrt(delta))/(2*a)
				x1 = (d+f*y1)/(-e)
				x2 = (d+f*y2)/(-e)
				intersect.append([x1,y1])
				intersect.append([x2,y2])
				return intersect

		elif f != 0: #Equation solvable in y
			a = (e**2)/(f**2)+1
			b = (2*d*e)/(f**2)+(2*cY1*e)/(f)-(2*cX1)
			c = -((r1**2)-(d**2)/(f**2)-(cY1**2)-(cX1**2)-(2*cY1*d)/(f))
			delta = (b**2)-(4*a*c)
			if delta < 0:
				return intersect

			elif delta==0:
				y = -b/(2*a)
				x = (d+e*y)/(-f)
				intersect.append([x,y])
				return intersect

			else:
				x1 = (-b+np.sqrt(delta))/(2*a)
				x2 = (-b-np.sqrt(delta))/(2*a)
				y1 = (d+e*x1)/(-f)
				y2 = (d+e*x2)/(-f)
				intersect.append([x1,y1])
				intersect.append([x2,y2])
				return intersect

		else: #Equation not solvable
			return intersect

def get_model_angles_from_endpoint(center, endpoint, lAlpha, lBeta):
	"""
	Returns the arm angles from the end point and the robot parameters (inverse model).

	Parameters
	----------
	center: tuple
		The center of the robot in the form (x (float), y (float))
	endpoint: tuple
		The fiber endpoint in the form (x (float), y (float))
	lAlpha: float
		The alpha arm length
	lBeta: float
		The beta arm length

	Returns
	-------
	list of angles:
		A list containing the alpha and beta arm angles [alphaAngle (float), betaAngle (float)].

	"""

	c1 = (center[1], center[0])
	c2 = [endpoint[1], endpoint[0]]
	r1 = lAlpha
	r2 = lBeta
	intersect = intersect_circles(c1,r1,c2,r2) #Get all the possible mid points (alpha-beta attachment point)

	angles = []
	for midpoint in intersect:
		alpha = np.arctan2(midpoint[1]-c1[1], midpoint[0]-c1[0])
		gamma = np.arctan2(c2[1]-midpoint[1], c2[0]-midpoint[0])
		beta = gamma-alpha

		angles.append([np.mod(alpha, 2*np.pi),np.mod(beta, 2*np.pi)])

	return angles

def get_closest(guessesList, targetList):
	"""
	Returns the closest element to a target from a list of guesses.

	The target can be multidimentionnal and it will return the closest point using arithmetic distance.

	Parameters
	----------
	guessesList: list
		The list containing the guesses. Each element is a list of floats with the same length as the target list (same dimension)
	targetList: list of floats
		The list containing the target coordinates

	Returns
	-------
	float: The closest guess

	"""

	dist = []
	# print((guessesList, targetList))
	for guess in guessesList:
		sumDist = 0
		for i in range(0, len(targetList)):
			sumDist += (targetList[i]-guess[i])**2
		dist.append(np.sqrt(sumDist/len(targetList)))
	if len(dist)<=0:
		return []
	bestGuess = guessesList[np.argmin(dist)]

	return bestGuess

def get_closest_angle(guessesList, targetList):
	"""
	Returns the closest element to a target from a list of guesses.

	The target can be multidimentionnal and it will return the closest point using arithmetic distance.\n
	Each guess angle will first be adapted to be in the same 360 degrees as the target angle. 

	Parameters
	----------
	guessesList: list
		The list containing the guesses. Each element is a list of floats with the same length as the target list (same dimension)
	targetList: list of floats
		The list containing the target coordinates

	Returns
	-------
	float: The closest guess

	"""

	for i in range(0, len(guessesList)):
		for j in range(0, len(targetList)):
			while targetList[j]-guessesList[i][j]<-np.pi:
				guessesList[i][j] -= 2*np.pi
			while targetList[j]-guessesList[i][j]>np.pi:
				guessesList[i][j] += 2*np.pi

	return get_closest(guessesList, targetList)

def isInCircle(coordinate,center,r):
	"""
	Returns True is the point is included in the circle's domain

	Parameters
	----------
	coordinate: tuple
		The point coordinates in the form (x (float), y (float))
	center: tuple
		The center of the circle in the form (x (float), y (float))
	r: float
		The radius of the circle
	
	Returns
	-------
	bool:
		True if the point is included in the circle's domain (perimeter included), False otherwise
	"""

	return bool(dist(coordinate, center) <= r)

def create_circular_mask(height, width, center, radius):
	"""
	Returns a circular mask for a 2D image.

	An array of "height" rows and "width" columns will be created and filled with booleans.\n
	The boolean will be True if the array's coordinate is included in the circle's area and False otherwise.

	Parameters
	----------
	height: float
		The number of rows of the mask
	width: float
		The number of columns of the mask
	center: tuple
		The center of the circle in the form (x (float), y (float))
	r: float
		The radius of the circle
	
	Returns
	-------
	np.ndarray:
		An array of boolean values. The boolean will be True if the array's coordinate is included in the circle's area and False otherwise.

	"""

	Y, X = np.ogrid[:height, :width]
	square_dist_from_center = (X - center[0])**2 + (Y-center[1])**2

	mask = square_dist_from_center <= radius**2
	return mask

def computeValidSoftROI(image, camMaxX, camMaxY, validityCenter, validityRadius):
	"""
	Computes the circular Region Of Interest (ROI) of an image

	It will crop the image to the absolute minimal size (either circle edge or image edge) and also apply a circular mask on it.\n
	Any pixel outside the circle is set to 0.
	
	Parameters
	----------
	image: np.ndarray
		A 2D array containing the grayscale value of each pixel in the image
	camMaxX: int
		The maximal X coordinate (width) of an image
	camMaxY: int
		The maximal Y coordinate (height) of an image
	validityCenter: tuple
		The center of the validity circle in the form (x (float), y (float))
	validityRadius: float
		The radius of the validity circle. If set to PC_IMAGE_GET_ALL_ROI, no change is made to the image

	Returns
	-------
	tuple: image, x_min, y_min
		image (np.ndarray): The cropped and masked image\n
		x_min: The new minimal x coordinate. It represents the number of columns cut at the left of the image.\n
		x_min: The new minimal y coordinate. It represents the number of rows cut at the top of the image.

	"""

	if validityRadius == DEFINES.PC_IMAGE_GET_ALL_ROI:
		return image, 0, 0
	else:
		x_min = int(validityCenter[0]-validityRadius)
		x_max = int(validityCenter[0]+validityRadius)
		y_min = int(validityCenter[1]-validityRadius)
		y_max = int(validityCenter[1]+validityRadius)

		#crop ROI while it is not in the image, up to the minimal window allowed
		if x_min < 0:
			x_min = 0
		if x_max-x_min < 1:
			x_max = x_min+1
		if x_max > camMaxX:
			x_max = camMaxX
		if x_max-x_min < 1:
			x_min = x_max-1
		if y_min < 0:
			y_min = 0
		if y_max-y_min < 1:
			y_max = y_min+1
		if y_max > camMaxY:
			y_max = camMaxY
		if y_max-y_min < 1:
			y_min = y_max-1

		image = image[	y_min:y_max,\
						x_min:x_max]

		validityCenter = (validityCenter[0]-x_min, validityCenter[1]-y_min) #Shift the circle center in the new image shape

		circularMask = create_circular_mask(image.shape[0], image.shape[1], validityCenter, validityRadius)
		image[~circularMask] = 0

		return image, x_min, y_min

def cropImage(image, ROI, camMaxX, camMaxY):
	"""
	Computes the rectangular Region Of Interest (ROI) of an image

	It will crop the image to the absolute minimal size. If the crop would go outside the image edges,
	it will be narrowed to these edges.
	
	Parameters
	----------
	image: np.ndarray
		A 2D array containing the grayscale value of each pixel in the image
	ROI: tuple
		Index 1 - CenterX (uint): The x coordinate of the center of the ROI\n
		Index 2 - CenterY (uint): The y coordinate of the center of the ROI\n
		Index 3 - Width (uint): The new desired image width\n
		Index 4 - Height (uint): The new desired image height
	camMaxX: int
		The maximal X coordinate (width) of an image
	camMaxY: int
		The maximal Y coordinate (height) of an image
	
	Returns
	-------
	tuple: image, x_min, y_min
		image (np.ndarray): The cropped image\n
		x_min: The new minimal x coordinate. It represents the number of columns cut at the left of the image.\n
		x_min: The new minimal y coordinate. It represents the number of rows cut at the top of the image.

	"""

	x_min = int(ROI[0]-ROI[2]/2)
	x_max = int(ROI[0]+ROI[2]/2)
	y_min = int(ROI[1]-ROI[3]/2)
	y_max = int(ROI[1]+ROI[3]/2)

	#crop ROI
	if x_min < 0:
		x_min = 0
	if x_max-x_min < 1:
		x_max = x_min+1
	if x_max > camMaxX:
		x_max = camMaxX
	if x_max-x_min < 1:
		x_min = x_max-1
	if y_min < 0:
		y_min = 0
	if y_max-y_min < 1:
		y_max = y_min+1
	if y_max > camMaxY:
		y_max = camMaxY
	if y_max-y_min < 1:
		y_min = y_max-1

	image = copy.deepcopy(image[	y_min:y_max,\
									x_min:x_max])

	return image,x_min,y_min

def nanrms(data):
	"""Returns the RMS value of the data, excluding any np.nan value"""

	data = np.ravel(data)
	if len(data)>0:
		if np.count_nonzero(~np.isnan(data))>0:
			return np.sqrt(np.nanmean(data**2))
		else:
			return np.nan
	else:
		return 0

def rms(data):
	"""Returns the RMS value of the data"""

	data = np.ravel(data)
	if len(data)>0:
		return np.sqrt(np.mean(data**2))
	else:
		return 0

def rms_err(data):
	"""Returns the RMS error of the data. The error of a point is its absolute difference with the mean of the data."""

	data = np.ravel(data)
	meanData = np.nanmean(data)
	if len(data)>0:
		return np.sqrt(np.nanmean((data-meanData)**2))
	else:
		return 0

def get_endpoint(centerX,centerY,lAlpha,lBeta,alphaAngle,betaAngle):
	"""
	Returns the theoretical endpoint of the robot given its parameters (direct model).

	Parameters
	----------
	centerX: float
		The X center of the robot
	centerY: float
		The Y center of the robot
	lAlpha: float
		The alpha arm length
	lBeta: float
		The beta arm length
	alphaAngle: float
		The alpha arm angle
	betaAngle: float
		The beta arm angle

	Returns
	-------
	tuple: targetX, targetY
		targetX (float): the x coordinate of the endpoint\n
		targetY (float): the y coordinate of the endpoint

	"""

	alphaAngle -= np.pi/2
	targetX = 	(np.cos(alphaAngle)*lAlpha+\
				np.cos(alphaAngle+betaAngle)*lBeta+\
				centerX)

	targetY = 	(-np.sin(alphaAngle)*lAlpha-\
				np.sin(alphaAngle+betaAngle)*lBeta+\
				centerY)

	return targetX, targetY

def _optimized_mean(data):
	data = data[~np.isnan(data)]
	estimate = np.nanmean(data)

	if np.isnan(estimate):
		return np.nan

	errorFunc = lambda value: np.sqrt((data-value)**2)
	result, success = optimize.leastsq(errorFunc, estimate, ftol = DEFINES.MM_MODEL_FIT_OPT_TOLERANCE, maxfev = DEFINES.MM_MODEL_FIT_OPT_MAX_F_EV)

	return result

def model_error(optParams,alphaCommand,betaCommand,alphaIterpolator,betaIterpolator,xData,yData,getFullOutput = False):
	"""
	Takes a set of robot parameters and computes the resulting model errors for a given dataset.

	Parameters
	----------
	optParams: tuple (centerX, centerY, lAlpha, lBeta, offsetAlpha, offsetBeta)
		The optimization parameters.\n
		centerX (float): The X center of the positioner\n
		centerY (float): The Y center of the positioner\n
		lAlpha (float): The alpha arm length\n
		lBeta (float): The beta arm length\n
		offsetAlpha (float): The alpha offset\n
		offsetBeta (float): The beta offset
	alphaCommand: np.ndarray
		The alpha motor commands for each point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	betaCommand: np.ndarray
		The beta motor commands for each point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	alphaIterpolator: scipy.interpolate.interp1d
		This is the iterpolator to return the corrected alpha angle (angle + non-linearity + offset) from any desired command angle
	betaIterpolator:  scipy.interpolate.interp1d
		This is the iterpolator to return the corrected beta angle (angle + non-linearity + offset) from any desired command angle
	xData: np.ndarray
		The X measurements at each commanded point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	yData: np.ndarray
		The Y measurements at each commanded point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	getFullOutput: bool, optional
		Set this to True to get the full output

	Returns
	-------
	resErrors: np.ndarray (if getFullOutput is False)
		The absolute errors of each point in the dataset in a 1D array
	tuple: resErrors, errorX, errorY  (if getFullOutput is True)
		resErrors (np.ndarray): the absolute error of each point in the dataset in a 4D array matching the dataset structure\n
		errorX(np.ndarray): the X error of each point in the dataset in a 4D array matching the dataset structure\n
		errorY(np.ndarray): the Y error of each point in the dataset in a 4D array matching the dataset structure

	"""

	(centerX,centerY,lAlpha,lBeta,offsetAlpha,offsetBeta) = optParams
	# print((centerX,centerY,lAlpha,lBeta,offsetAlpha,offsetBeta))

	#iterate to get all the errors
	valuesToRemove = np.isnan(xData)
	(nbRepetitions,nbStartingPoints,nbAxes,nbSteps) = xData.shape
	totalNbPoints = nbRepetitions*nbStartingPoints*nbAxes*nbSteps
	resErrors = np.full((nbRepetitions,nbStartingPoints,nbAxes,nbSteps),np.nan)
	errorX = np.full((nbRepetitions,nbStartingPoints,nbAxes,nbSteps),np.nan)
	errorY = np.full((nbRepetitions,nbStartingPoints,nbAxes,nbSteps),np.nan)

	for repetition in range(0,nbRepetitions):
		for startingPoint in range(0,nbStartingPoints):
			for axis in range(0,nbAxes):
				for step in range(0,nbSteps):
					if not valuesToRemove[repetition,startingPoint,axis,step]:
						#get model angle
						alphaAngle = alphaIterpolator(alphaCommand[startingPoint,axis,step])+offsetAlpha
						betaAngle = betaIterpolator(betaCommand[startingPoint,axis,step])+offsetBeta

						(targetX,targetY) = get_endpoint(centerX,centerY,lAlpha,lBeta,alphaAngle,betaAngle)
						
						errorX[repetition,startingPoint,axis,step] = targetX-xData[repetition,startingPoint,axis,step]
						errorY[repetition,startingPoint,axis,step] = targetY-yData[repetition,startingPoint,axis,step]

						resErrors[repetition,startingPoint,axis,step] = np.sqrt(errorX[repetition,startingPoint,axis,step]**2+errorY[repetition,startingPoint,axis,step]**2)

	if getFullOutput:
		return resErrors,errorX,errorY
	else:
		resErrors = resErrors[~np.isnan(resErrors)]
		return np.ravel(resErrors)

def mean_model_error(optParams,alphaCommand,betaCommand,alphaIterpolator,betaIterpolator,xData,yData):
	"""Returns the mean model error of a given set of robot parameters. See the function model_error for parameters"""

	allErrors = model_error(optParams,alphaCommand,betaCommand,alphaIterpolator,betaIterpolator,xData,yData,False)
	return np.nanmean(allErrors)*1000

def rms_model_error(optParams,alphaCommand,betaCommand,alphaIterpolator,betaIterpolator,xData,yData):
	"""Returns the mean model error of a given set of robot parameters. See the function model_error for parameters"""

	allErrors = model_error(optParams,alphaCommand,betaCommand,alphaIterpolator,betaIterpolator,xData,yData,False)
	return nanrms(allErrors)*1000

def optimize_model(centerX,centerY,lAlpha,lBeta,offsetAlpha,offsetBeta,alphaCommand,betaCommand,alphaMeasures,betaMeasures,xData,yData):
	"""
	Optimizes the positioner model

	It takes the measurements and the commands as input, along with a set of initial parameters. \n
	It will then perform a least square optimization on the data and return the fitted parameters.

	Parameters
	----------
	centerX: float
		The X center of the positioner initial value
	centerY: float
		The Y center of the positioner initial value
	lAlpha: float
		The alpha arm length initial value
	lBeta: float
		The beta arm length initial value
	offsetAlpha: float
		The alpha offset initial value
	offsetBeta: float
		The beta offset initial value
	alphaCommand: np.ndarray
		The alpha motor commands for each point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	betaCommand: np.ndarray
		The beta motor commands for each point.
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	alphaMeasures: np.ndarray
		The alpha arm angle measures for each point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	betaMeasures: np.ndarray
		The beta arm angle measures for each point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	xData: np.ndarray
		The X measurements at each commanded point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target
	yData: np.ndarray
		The Y measurements at each commanded point.\n
		It has the following structure:\n
		Index 0 - startingPoint: The n-th arm configuration \n
		Index 1 - axis: The axis that is moving\n
		Index 2 - step: The step\n
		Index 3 - motorIndex: The motor to which the command is destined\n
		(0) The alpha motor target\n
		(1) The beta motor target

	Returns
	-------
	params: tuple
		The optimized parameters.\n
		Index 0 - centerX (float): The X center of the positioner optimized value\n
		Index 1 - centerY (float): The Y center of the positioner optimized value\n
		Index 2 - lAlpha (float): The alpha arm length optimized value\n
		Index 3 - lBeta (float): The beta arm length optimized value\n
		Index 4 - offsetAlpha (float): The alpha offset optimized value\n
		Index 5 - offsetBeta (float): The beta offset optimized value

	"""

	offsetAlpha=np.mod(offsetAlpha+np.pi,2*np.pi)-np.pi
	offsetBeta=np.mod(offsetBeta+np.pi,2*np.pi)-np.pi

	params = (centerX,centerY,lAlpha,lBeta,offsetAlpha,offsetBeta)
	# print(params)
	
	nbSteps = (alphaMeasures.shape)[2]
	meanAlphaMeasures = np.full((nbSteps),np.nan)
	meanBetaMeasures = np.full((nbSteps),np.nan)
	meanAlphaCommand = np.full((nbSteps),np.nan)
	meanBetaCommand = np.full((nbSteps),np.nan)

	for step in range(0,nbSteps):
		meanAlphaMeasures[step] = np.nanmean(np.ravel(alphaMeasures[:,:,step]))
		meanBetaMeasures[step] = np.nanmean(np.ravel(betaMeasures[:,:,step]))
		meanAlphaCommand[step] = np.nanmean(np.ravel(alphaCommand[0,DEFINES.PARAM_AXIS_ALPHA,step]))
		meanBetaCommand[step] = np.nanmean(np.ravel(betaCommand[0,DEFINES.PARAM_AXIS_BETA,step]))
	
	meanAlphaCommand = meanAlphaCommand[~np.isnan(meanAlphaMeasures)]
	meanBetaCommand = meanBetaCommand[~np.isnan(meanBetaMeasures)]
	meanAlphaMeasures = meanAlphaMeasures[~np.isnan(meanAlphaMeasures)]
	meanBetaMeasures = meanBetaMeasures[~np.isnan(meanBetaMeasures)]

	# print((meanAlphaMeasures,meanAlphaCommand))
	# print((meanBetaMeasures,meanBetaCommand))

	#construct the alpha and beta approximators
	alphaIterpolator = interpolate.interp1d(meanAlphaCommand, meanAlphaMeasures, kind='linear', fill_value='extrapolate')
	betaIterpolator = interpolate.interp1d(meanBetaCommand, meanBetaMeasures, kind='linear', fill_value='extrapolate')

	modelFit = rms_model_error(params,alphaCommand,betaCommand,alphaIterpolator,betaIterpolator,xData,yData)
	
	if len(~np.isnan(np.ravel(xData))) >= len(params):
		degenerated_error = lambda params: model_error(params,alphaCommand,betaCommand,alphaIterpolator,betaIterpolator,xData,yData,False)
		params, success = optimize.leastsq(degenerated_error, params, ftol = DEFINES.MM_MODEL_FIT_OPT_TOLERANCE, maxfev = DEFINES.MM_MODEL_FIT_OPT_MAX_F_EV)
	else:
		log.message(DEFINES.LOG_MESSAGE_PRIORITY_WARNING,1,f'Optimization skipped. Not enough data available.')

	params = np.asarray(params)
	params[4] = np.mod(params[4]+np.pi,2*np.pi)-np.pi
	params[5] = np.mod(params[5]+np.pi,2*np.pi)-np.pi #adapt offsets between -pi and pi

	modelFit = rms_model_error(params,alphaCommand,betaCommand,alphaIterpolator,betaIterpolator,xData,yData)
	
	return params

def threshold(data, min_val, max_val):
	"""
	Borns the values in an array between two borns.

	Parameters
	----------
	data: np.ndarray or list
		The data to be borned, in any shape
	min_val: float
		The floor born. Any value smaller than that will be adjusted to match it.
	max_val: float
		The ceilig born. Any value larger than that will be adjusted to match it.

	Returns
	-------
	bornedData: type matches the input data
		The new array with all content borned between min_value and max_value.

	"""

	data[data>max_val] = max_val
	data[data<min_val] = min_val
	return data

def threshold1D(data, min_val, max_val):
	"""
	Borns the value between two borns.

	Parameters
	----------
	data: float
		The value to be borned
	min_val: float
		The floor born. Any value smaller than that will be adjusted to match it.
	max_val: float
		The ceilig born. Any value larger than that will be adjusted to match it.

	Returns
	-------
	bornedData: float
		The new value borned between min_value and max_value.

	"""

	if data>max_val:
		data=max_val
	elif data<min_val:
		data=min_val
	return data

def circle(centerX, centerY, radius):
	"""
	Returns a lambda function mapping the angle on a circle to x,y coordinates
	
	Parameters
	----------
	centerX: float
		The X center of the circle's center
	centerY: float
		The Y center of the circle's center
	radius: float
		The radius of the circle

	Returns
	-------
		A lambda function f returning a tuple of coordinates. (X,Y) = f(angle)

	"""

	return lambda alpha: (	centerX + np.cos(np.deg2rad(alpha))*radius,\
							centerY - np.sin(np.deg2rad(alpha))*radius)

def distToCircle(centerX, centerY, radius):
	"""
	Returns a lambda function giving the distance between a point and the circle perimeter.

	Any point inside the circle will give a negative distance.

	Parameters
	----------
	centerX: float
		The X center of the circle
	centerY: float
		The Y center of the circle
	radius: float
		The radius of the circle

	Returns
	-------
	A lambda function f giving the distance d between a point (x,y) and the circle perimeter. d = f(x,y)

	"""

	return lambda x,y: np.sqrt((centerX-x)**2+(centerY-y)**2)-radius

def gaussian(height, center_x, center_y, width_x, width_y, n, angle = 0):
	"""
	Returns a lambda function giving the height (value) of a 2D generalized gaussian at a given point.

	Parameters
	----------
	height: float
		The maximal height of the gaussian
	center_x: float
		The X center of the gaussian. It represents the position of the peak.
	center_y: float
		The Y center of the gaussian. It represents the position of the peak.
	width_x: float
		The X sigma value. It represents the spread in the X direction.
	width_y: float
		The Y sigma value. It represents the spread in the Y direction.
	n: float
		The gaussian shape parameter (gaussian order)
	angle: float
		The gaussian orientation, in degrees

	Returns
	-------
	A lambda function f giving the height (value) h of a 2D gaussian at a given point (x,y). h = f(x,y)

	"""

	sqr_width_x = float(width_x)**2
	sqr_width_y = float(width_y)**2

	cos_angle = np.cos(angle*np.pi/180)
	sin_angle = np.sin(angle*np.pi/180)

	a = cos_angle**2/(2*sqr_width_x)+sin_angle**2/(2*width_y)
	b = -sin_angle/(4*sqr_width_x)+sin_angle/(4*width_y)
	c = sin_angle**2/(2*sqr_width_x)+cos_angle**2/(2*width_y)

	return lambda x,y: (height*np.exp(-(
				a*(x-center_x)**2+2*b*(x-center_x)*(y-center_y)+c*(y-center_y)**2))).astype(np.longdouble)

	# return lambda x,y: (height*np.exp(
	# 			-((((center_x-x)/width_x)**2+(abs(center_y-y)/width_y)**2)/2)**n)).astype(np.longdouble)

def moments(data,Xin,Yin):
	"""
	Returns the estimated parameters for a gaussian distribution using the moments of the image.

	Parameters
	----------
	data : np.ndarray
		2D array containing the image pixel values as floats
	Xin : np.ndarray
		2D array containing the X coordinate value of each pixel as floats
	Yin : np.ndarray
		2D array containing the Y coordinate value of each pixel as floats

	Returns
	-------
	tuple: height, center_x, center_y, width_x, width_y, n
		height (float) - The estimated maximal height of the gaussian\n
		center_x (float) - The estimated X center of the gaussian. It represents the position of the peak.\n
		center_y (float) - The estimated Y center of the gaussian. It represents the position of the peak.\n
		width_x (float) - The estimated X sigma value. It represents the spread in the X direction.\n
		width_y (float) - The estimated Y sigma value. It represents the spread in the Y direction.\n
		n (float) - The estimated gaussian order. Always 1
		angle (float) - The estimated gaussian orientation. Always 0

	"""

	total = np.sum(data)
	if total == 0:
		return None
	X, Y = [Xin,Yin]
	center_x = np.sum((X*data)/total)
	center_y = np.sum((Y*data)/total)
	if int(center_y) >= (data.shape)[1]:
		return None
	col = data[:, int(center_y)]
	width_x = np.sqrt(np.abs((np.arange(col.size)-center_y)**2*col).sum()/col.sum())
	if int(center_x) >= (data.shape)[0]:
		return None
	row = data[int(center_x), :]
	width_y = np.sqrt(np.abs((np.arange(row.size)-center_x)**2*row).sum()/row.sum())
	height = data.max()
	n = 1
	angle = 0
	return height, center_x, center_y, width_x, width_y, n, angle

def fitgaussian(data,Xin,Yin,data_min,data_max,optimizerTolerance,estimate):
	"""
	Fits a 2D gaussian on an greyscale image and returns its parameters.

	It will perform a least square optimization on the given estimate.

	Parameters
	----------
	data : np.ndarray
		2D array containing the image pixel values as floats
	Xin : np.ndarray
		2D array containing the X coordinate value of each pixel as floats
	Yin : np.ndarray
		2D array containing the Y coordinate value of each pixel as floats
	data_min: float
		The minimal value of a pixel. Any pixel value higher will be adapted to match data_min
	data_max: float
		The maximal value of a pixel. Any pixel value higher will be adapted to match data_max
	optimizerTolerance: float
		The optimizer's stop condition. See scipy.optimize.leastsq function for more details.
	estimate: tuple
		Index 0 (float) - The estimated maximal height of the gaussian\n
		Index 1 (float) - The estimated X center of the gaussian. It represents the position of the peak.\n
		Index 2 (float) - The estimated Y center of the gaussian. It represents the position of the peak.\n
		Index 3 (float) - The estimated X sigma value. It represents the spread in the X direction.\n
		Index 4 (float) - The estimated Y sigma value. It represents the spread in the Y direction.\n
		Index 5 (float) - The estimated gaussian shape parameter

	Returns
	-------
	optimized: tuple
		Index 0 (float) - The optimized maximal height of the gaussian\n
		Index 1 (float) - The optimized X center of the gaussian. It represents the position of the peak.\n
		Index 2 (float) - The optimized Y center of the gaussian. It represents the position of the peak.\n
		Index 3 (float) - The optimized X sigma value. It represents the spread in the X direction.\n
		Index 4 (float) - The optimized Y sigma value. It represents the spread in the Y direction.\n
		Index 5 (float) - The optimized gaussian shape parameter
	
	"""

	Xin = Xin.astype(np.longdouble)
	Yin = Yin.astype(np.longdouble)
	data = data.astype(np.longdouble)
	dataShapeX, dataShapeY= data.shape
	errorfunction = lambda p: np.ravel(threshold(gaussian(*p)(Xin,Yin),data_min,data_max) - data)
	
	if dataShapeX*dataShapeY >= len(estimate):
		estimate, success = optimize.leastsq(errorfunction, estimate, ftol = optimizerTolerance)

	estimate = np.asarray(estimate)

	return estimate

def get_img_ID(image_ID):
	"""
	Uncompresses an image ID.

	The image ID is containing the information on when the image was 
	taken in the test's progression and to which positioner it belongs.\n
	See the return values for more details.

	Parameters
	----------
	image_ID: int
		The comressed image ID generated using generate_img_ID

	Returns
	-------
	tuple: benchSlot, repetition, startingPoint, axis, step, direction, centroidType
		benchSlot (int) - The testbench slot the image was shot from\n
		repetition (int) - The repetition at which the image was taken\n
		startingPoint (int) - The startingPoint at which the image was taken\n
		axis (int) - The axis that was moveing when the image was taken\n
		step (int) - The step at which the image was taken\n
		direction (int) - The direction at which the image was taken\n
		centroidType (int) - The camera type that took the picture (MM_IMG_ID_XY_IDENTIFIER or MM_IMG_ID_TILT_IDENTIFIER)

	"""

	centroidType = 		(image_ID & MM_IMG_ID_BITMASK_FOR_CENTROID_TYPE)	>> MM_IMG_ID_BITSHIFT_FOR_CENTROID_TYPE
	direction = 		(image_ID & MM_IMG_ID_BITMASK_FOR_DIRECTION)		>> MM_IMG_ID_BITSHIFT_FOR_DIRECTION
	axis = 				(image_ID & MM_IMG_ID_BITMASK_FOR_AXIS)				>> MM_IMG_ID_BITSHIFT_FOR_AXIS
	step = 				(image_ID & MM_IMG_ID_BITMASK_FOR_STEP)				>> MM_IMG_ID_BITSHIFT_FOR_STEP
	repetition = 		(image_ID & MM_IMG_ID_BITMASK_FOR_REPETITION)		>> MM_IMG_ID_BITSHIFT_FOR_REPETITION
	startingPoint = 	(image_ID & MM_IMG_ID_BITMASK_FOR_STARTING_POINT)	>> MM_IMG_ID_BITSHIFT_FOR_STARTING_POINT
	benchSlot = 		(image_ID & MM_IMG_ID_BITMASK_FOR_BENCH_SLOT)		>> MM_IMG_ID_BITSHIFT_FOR_BENCH_SLOT

	return benchSlot, repetition, startingPoint, axis, step, direction, centroidType

def generate_img_ID(benchSlot, repetition, startingPoint, axis, step, direction, centroidType):
	"""
	Compresses a bunch of numbers into an image ID

	The image ID is containing the information on when the image was 
	taken in the test's progression and to which positioner it belongs.
	Each variable has a limited amount of bits and it will compress these to one unique integer.
	
	Parameters
	----------
	benchSlot: int
		The testbench slot the image was shot from. Maximal value is 2**MM_IMG_ID_BITS_FOR_BENCH_SLOT-1
	repetition: int
		The repetition at which the image was taken. Maximal value is 2**MM_IMG_ID_BITS_FOR_REPETITION-1
	startingPoint: int
		The startingPoint at which the image was taken. Maximal value is 2**MM_IMG_ID_BITS_FOR_STARTING_POINT-1
	axis: int
		The axis that was moveing when the image was taken. Maximal value is 2**MM_IMG_ID_BITS_FOR_AXIS-1
	step: int
		The step at which the image was taken. Maximal value is 2**MM_IMG_ID_BITS_FOR_STEP-1
	direction: int
		The direction at which the image was taken. Maximal value is 2**MM_IMG_ID_BITS_FOR_DIRECTION-1
	centroidType: int
		The camera type that took the picture (MM_IMG_ID_XY_IDENTIFIER or MM_IMG_ID_TILT_IDENTIFIER). Maximal value is 2**MM_IMG_ID_BITS_FOR_CENTROID_TYPE-1


	Returns
	-------
	imageID: int
		The comressed image ID
		
	"""

	return 	np.int64(	(centroidType 	<< MM_IMG_ID_BITSHIFT_FOR_CENTROID_TYPE) +\
						(direction		<< MM_IMG_ID_BITSHIFT_FOR_DIRECTION) + \
						(axis			<< MM_IMG_ID_BITSHIFT_FOR_AXIS) + \
						(step			<< MM_IMG_ID_BITSHIFT_FOR_STEP) + \
						(repetition		<< MM_IMG_ID_BITSHIFT_FOR_REPETITION) + \
						(startingPoint 	<< MM_IMG_ID_BITSHIFT_FOR_STARTING_POINT) + \
						(benchSlot 		<< MM_IMG_ID_BITSHIFT_FOR_BENCH_SLOT))
						
def get_ETA(tStart, completion):
	"""
	Returns the estimated end time using the start time and the current completion.

	It performs a linear extrapolation to get the final time

	Parameters
	----------
	tStart: time.Time
		The starting time using the time module
	completion: float
		The completion percentage, bewteen 0 (excluded) and 1 (included)

	Returns
	-------
	tuple: tRemaining, days, hours, minutes, seconds
		tRemaining (float) - The estimated total number of seconds remaining\n
		days (int) - The estimated number of days remaining\n
		hours (int) - The estimated number of hours remaining, days exculded\n
		minutes (int) - The estimated number of minutes remaining, days and hours exculded\n
		seconds (float) - The estimated number of seconds remaining, days, hours and minutes excluded

	"""

	tNow = time.time()
	tRemaining = (tNow-tStart)/completion-(tNow-tStart)
	(days, hours, minutes, seconds) = decompose_time(tRemaining)

	return tRemaining, days, hours, minutes, seconds

def decompose_time(time):
	"""
	Returns a decomposition of the time in days, hours, minutes and seconds

	Parameters
	----------
	time: float
		The time to decompose, in seconds

	Returns
	-------
	tuple: days, hours, minutes, seconds
		days (int) - The number of days\n
		hours (int) - The number of hours, days exculded\n
		minutes (int) - The number of minutes, days and hours exculded\n
		seconds (float) - The number of seconds, days, hours and minutes excluded

	"""

	days, restSeconds = divmod(time,24*60*60)
	hours, restSeconds = divmod(restSeconds,60*60)
	minutes, seconds = divmod(restSeconds,60)

	return int(days), int(hours), int(minutes), float(seconds)

def _main():
	circle1 = (32.95,42.87,7.45)
	xData = circle(circle1[0],circle1[1],circle1[2])(np.linspace(0,10,15, endpoint=False))[0]
	yData = circle(circle1[0],circle1[1],circle1[2])(np.linspace(0,10,15, endpoint=False))[1]
	xData = list(xData)
	yData = list(yData)

	xData.append(15)
	yData.append(25)
	valuesToDrop = [False for i in range(0,len(xData))]

	center = get_circle_center_approx(xData, yData)

	plt.scatter(center[0], center[1], color = 'red')
	plt.draw()
	plt.pause(1e-17)
	print(center)
	dist = [np.sqrt((center[0]-xData[i])**2 + (center[0]-yData[i])**2) for i in range(0,len(xData))]
	zn = np.abs(stats.zscore(dist))

	# a = generate_img_ID(1,25,34,1,43523,1,0)
	# print(get_img_ID(a))
	# print(get_circumcenter((0,2),(2,0),(0,0)))
	# circle1 = (32.95,42.87,7.45)
	# print(fit_circle((circle(circle1[0],circle1[1],circle1[2])(np.linspace(0,2,3, endpoint=False)))[0],(circle(circle1[0],circle1[1],circle1[2])(np.linspace(0,2,3, endpoint=False)))[1]))

	# print(get_ETA(time.time()-86345.22,0.5))
	# print(get_endpoint(25,40,10,10,np.pi/2,np.pi/2)) #x, y, l1, l2, a, b

if __name__ == '__main__':
	_main()