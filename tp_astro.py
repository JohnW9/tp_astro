import classCamera
import DEFINES
import classConfig
import positioner
from serial.tools import list_ports
import can


def tp_init(comBus = None):
    cam = None
    pos = None
    bus = None

    # prepare camera and set auto exposure
    exposure_time = 1000

    cam = classCamera.Camera(cameraType = DEFINES.PC_CAMERA_TYPE_XY)
    cam.setMaxROI()
    exposure_time = cam.getOptimalExposure(exposure_time)
    config = classConfig.Config()
    cam.setDistortionCorrection(config)

    # check communication handler for can bus
    if comBus is None:
        comPorts = list(list_ports.comports())
        for port_no, description, device in comPorts:
            if 'USB' in description:
                try:
                    bus = can.interface.Bus(port_no, bustype='slcan', ttyBaudrate=921600, bitrate=1000000)
                    print('communication bus at ' + port_no)
                except:
                    print ('no commucation bus found')
    else:
        try:
            bus = can.interface.Bus(port_no, bustype='slcan', ttyBaudrate=921600, bitrate=1000000)
            print('communication bus at ' + comBus)
        except:
                print('no communication bus')

    if bus is not None:
        broadcast = positioner.Positioner(bus, 0)
        id = broadcast.get_id()
        pos = positioner.Positioner(bus, id)
        pos.set_precision_mode_off()
        pos.set_speed(3000, 3000)
    return cam, pos