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
# Alpha axis will move to 30° and beta to 90
pos.goto_absolute(30,90)
```
* Wait for the move to finish before continuing
```python
pos.wait_move()
```
* Change the angular speed of each axis, in RPM on the motor side (speed limits: [1000, 3000] RPM)
```python
# Speed for the next move will be changed to 2000 RPM for both axis
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
cam, pos = tp.tp_init()
pos.goto_absolute(30,90)
pos.wait_move()
x_centroid, y_centroid = cam.getCentroid()
print(x_centroid, y_centroid)
```

## Fit circle 

Finally you will also by provided the fit_circle function, i.e. determine the closest circle fit from a set of data points <br>
**Note:** the code is optimized to handle a large number of data points. So take at least 4 sample points for circle fitting, even though 3 are enough in theory.

```python
from miscmath import fit_circle
# /!\ xData, yData are np arrays of the collected data points coordinates /!\
center_x, center_y, radius = fit_circle(xData, yData)
```

<!--- 
## Perform a firmware upgrade
The following commands will perform a firmware upgrade on the device with ID 4 on an ixxat bus.
```python
import can
import positioner
canbus = can.interface.Bus(bustype='ixxat', channel=0, bitrate=1000000)
pos4 = positioner.Positioner(canbus, 4)
# get firmware version to make sure we are in bootloader
pos4.get_fw_version()
# make sure version is xx.80.zz
pos4.firmware_upgrade(r'sdssv_v2.bin')
# aditionnal checks can be done with status to make sure the new image was loaded and checksum was ok
```


## Send a trajectory
Send a trajectory and initiate move
```python
import can
import positioner
canbus = can.interface.Bus(bustype='ixxat', channel=0, bitrate=1000000)
pos4 = positioner.Positioner(canbus, 4)
# get firmware version to make sure we are main firmware
# make sure version is xx.02.zz
pos4.get_fw_version()
# init the datums otherwise positioner will refuse to move, LEDs will blink alternatively after init is done
pos4.initialize_datums()
# sets the trajectories (list of tupplies (degree, time [s])
alpha_traj = [(60, 5), (60, 10), (120, 20), (120, 30), (0, 45)]
beta_traj = [(0, 5), (60, 10), (60, 20), (120, 30), (0, 45)]
# send the trajectories
pos4.send_trajectory(alpha_traj, beta_traj)
# initiate the move
pos4.start_trajectory()
``` 
-->