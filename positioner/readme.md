# SDSS-V Positioner interface

This library allows communication with the SDSS-V Positioner though the CAN bus.

# Setup

In order to use it, python-can package should be installed.
Until the next release of the package (2.3.0), where the serial interface is going to be updated the following version is used for testing:
https://github.com/boris-wenzlaff/python-can/tree/serialcom_b

# Usage

The usage is pretty simple:
* import can and positioner
* make a CANbus interface
* setup a positioner with the can bus interface and the ID
* use it :-)

## Setup a can interface

To set up an interface, simply use the bus.can.interface with proper options.
for an ixxat device use:
```python
bus = can.interface.Bus(bustype='ixxat', channel=0, bitrate=1000000)
```
for a serial device with the usbcan adapter use:
```python
bus = can.interface.Bus('COM3', bustype='slcan', ttyBaudrate=921600, bitrate=1000000)
```

## Setup a positioner

Simply declare a positioner with the bus created earlier and the CAN ID.
```python
my_positioner = positioner.Positioner(bus, 4)
```
> The ID 0 can be used for broadcast commands.

# Examples

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


## Perform a manual move
Make a move.
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
# set the speed, both motors are 1000 rpm, by default speed is 0 so it won't move
pos4.set_speed(1000, 1000)
# send a position to where to move in degrees
pos4.goto_absolute(60, 60)
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

