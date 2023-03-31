# SDSS-V Positioner & Camera interface

This library allows communication with the SDSS-V Positioner though the CAN bus and with the Basler camera via USB.

# Setup

Python version: 3.7 or after
In order to use it the modules in requirement.txt must be available for the used python interpreter

Note for serial link between PC and positioner:
Until the next release of the package (2.3.0), where the serial interface is going to be updated the following version is used for testing:
https://github.com/boris-wenzlaff/python-can/tree/serialcom_b

# Usage

The usage is pretty simple:
* import tp_init() from tp_astro
* initialiaze CAN bus for positioner & USB link for camera (example below)
* use them :-)

<!--- ## Setup a can interface

To set up an interface, simply use the bus.can.interface with proper options.
for an ixxat device use:
```python
bus = can.interface.Bus(bustype='ixxat', channel=0, bitrate=1000000)
```
for a serial device with the usbcan adapter use:
```python
bus = can.interface.Bus('COM3', bustype='slcan', ttyBaudrate=921600, bitrate=1000000)
```
-->

<!---  ## Setup a positioner

Simply declare a positioner with the bus created earlier and the CAN ID.
```python
my_positioner = positioner.Positioner(bus, 4)
```
> The ID 0 can be used for broadcast commands.
-->
# Examples

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
* Move to a relative angular position from current position, say (alpha = 30°, beta = 90°)
```python
# Alpha axis will move 30° from current position and beta 90° from current pos
pos.goto_relative(30,90)
```
* Change the angular speed of each axis, in RPM on the motor side (speed limits: [1000, 3000] RPM)
```python
# Speed for the next move will be changed to 2000 RPM for both axis
pos.set_speed(2000,2000)
```
* Wait for the move to finish before continuing (totally ramdomly: useful for waiting for the fiber to get in position to acquire its centroid)
```python
pos.wait_move()
```

## Acquire data with camera

The camera allows you to acquire the center point cartesian coordinates of the dot of light on the robot 

```python
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

# Fit circle 

Finally you will also by provided the fit_circle function, i.e. determine the closest circle fit from a set of data points

```python
from miscmath import fit_circle
# xData, yData are np arrays of the collected data points coordinates
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