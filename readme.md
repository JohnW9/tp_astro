# Positioner & Camera interface

This library allows communication with the fiber positioner though the CAN bus and with the Basler camera via USB.

# Setup

Python version: 3.7 or after <br>
In order to use it the modules in *requirement.txt* must be available for the used python interpreter <br>
The [*Cameras* folder](https://drive.google.com/drive/folders/1heMNF-fCXqoXDatZ4CJjCBfjY_bv1eqs?usp=share_link) contains the calibration files for the cameras <br>
If not existing, create a folder *Cameras* in the *Config* folder and paste the calibration files from Drive.

# Usage

The usage is pretty simple:
* import tp_init() from tp_astro
* use *cam* & *pos* objects to acquire data and move around the robot :-)

# Examples

Here are all the python functions you need to get through this practicals <br>

## Setup communication for camera & positioner

Create cam and pos objects to call the functions needed to perform this practical
```python
import tp_astro as tp
cam, pos = tp.tp_init()
```

## Perform a manual move
Make a move.
Each axis can rotate from 0° to 360°. Once the pos object is created you can either:

* Move to an absolute angular position, say (alpha = 30°, beta = 90°)
```python
"""
Input: None
Output: robot moves to absolute angular positons
"""
pos.goto_absolute(30,90)
```
* IMPORTANT: Wait for the move to finish before continuing
```python
"""
Input: None
Output: code pauses executing until robot finishes its move
"""
pos.wait_move()
```
* Change the angular speed of each axis, in RPM on the motor side (speed limits: [1000, 3000] RPM)
```python
"""
Input: alpha speed [scalar in RPM], beta speed [scalar in RPM]
Output: speed set for next moves until new change
"""
pos.set_speed(2000,2000)
```
* Move to a relative angular position from current position, say (alpha = 30°, beta = 90°)
```python
"""
Input: alpha [scalar in °], beta [scalar in °]
Output: robot moves to position relative to current one
"""
pos.goto_relative(30,90)
```

## Acquire data with camera

The camera allows you to acquire the centroid of the fiber on the image coordinate 

```python
"""
Input: None
Output: x,y = coordinates [mm, mm] of centroid of fiber on image
"""
x_centroid, y_centroid = cam.getCentroid()
```

## Example of scripted data acquisition

This code snippet summarizes in one shot:
* Initialize commmunication  objects
* Make a move
* Get position of the fiber centroid
```python
import tp_astro as tp
cam, pos = tp.tp_init() # Initialize camera & robot communication
pos.goto_absolute(30,90) # Move robot to alpha = 30°, beta = 90°
pos.wait_move() # Wait for trajectory to finish before continuing
x_centroid, y_centroid = cam.getCentroid() # Acquire position of centroid on image
print(x_centroid, y_centroid)
```

## Fit circle 

Finally you will also by provided the fit_circle function, i.e. determine the closest circle fit from a set of data points <br>
**Note:** the code is optimized to handle a large number of data points. So take at least 4 sample points for circle fitting, even though 3 are enough in theory.

```python
from miscmath import fit_circle
"""
Input : - xData (np array), yData (np array) = x,y coordinate of sample points
Output: - center_x [mm], center_y [mm] = center of fitted circle
        - radius [mm] = radius of fitted circle
"""

center_x, center_y, radius = fit_circle(xData, yData)
```