from yumipy import YuMiRobot
from phoxipy import ColorizedPhoXiSensor
from phoxipy.phoxi_sensor import PhoXiSensor
from perception import ColorImage, RgbdImage, CameraIntrinsics, WebcamSensor
import os

if __name__ == "__main__": 
    y = YuMiRobot()
    pose = y.left.get_pose()
    print(pose)
    y.left.goto_pose_delta((0,0,0), (5,-5,-10))
    # camera = ColorizedPhoXiSensor("colorized_phoxi", )
    print("load intrinsics")
    phoxi_intr = CameraIntrinsics.load(os.path.join(os.path.dirname(__file__),'calib','phoxi', 'phoxi.intr'))
    print("Create Camera")
    camera = PhoXiSensor("2018-02-020-LC3", phoxi_intr)
    print("Start camera")
    camera.start()
    print("Read from Camera")
    camera.read()
