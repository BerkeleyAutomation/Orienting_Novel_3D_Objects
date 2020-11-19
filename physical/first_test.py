from yumipy import YuMiRobot
from phoxipy import ColorizedPhoXiSensor
from phoxipy.phoxi_sensor import PhoXiSensor
from perception import ColorImage, RgbdImage, CameraIntrinsics, WebcamSensor
from autolab_core import RigidTransform, YamlConfig, Logger
import os

def test_movement():
    y = YuMiRobot()
    pose = y.left.get_pose()
    print(pose)
    y.left.goto_pose_delta((0,0,0), (5,-5,-10))

def test_camera_phoxipy():
    print("load intrinsics")
    calib_dir = '/nfs/diskstation/calib/phoxi'
    phoxi_intr = CameraIntrinsics.load(os.path.join(calib_dir, 'phoxi.intr'))
    T_camera_world = RigidTransform.load(os.path.join(calib_dir, 'phoxi_to_world.tf'))

    print("Create Camera")
    camera = PhoXiSensor("1703005", phoxi_intr)
    # camera = ColorizedPhoXiSensor("colorized_phoxi", )
    print("Start camera")
    camera.start()
    print("Read from Camera")
    camera.read()

def test_camera():
    calib_dir = '/nfs/diskstation/calib/phoxi'
    phoxi_intr = CameraIntrinsics.load(os.path.join(calib_dir, 'phoxi.intr'))
    T_camera_world = RigidTransform.load(os.path.join(calib_dir, 'phoxi_to_world.tf'))
    phoxi_config = YamlConfig("cfg/tools/colorized_phoxi.yaml")
    sensor_name = 'phoxi'
    sensor_config = phoxi_config['sensors'][sensor_name]

    # logger.info('Ready to capture images from sensor %s' %(sensor_name))
    save_dir = "ros_phoxi"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)        
    
    # read params
    sensor_type = sensor_config['type']
    sensor_frame = sensor_config['frame']
    sensor = RgbdSensorFactory.sensor(sensor_type, sensor_config)
    sensor.start()

    # logger.info('Capturing depth image %d' %(frame))
    color, depth, _ = sensor.frames()
    return depth

# def test_grasp():
    # def start(self):
    #     """ Connect to the robot and reset to the home pose. """
    #     # iteratively attempt to initialize the robot
    #     initialized = False
    #     self.robot = None
    #     while not initialized:
    #         try:
    #             # open robot
    #             self.robot = YuMiRobot(debug=self.config['debug'],
    #                                    arm_type=self.config['arm_type'])                # reset the arm poses
    #             self.robot.set_z(self.zoning)
    #             self.robot.set_v(self.velocity)
    #             if self._reset_on_start:
    #                 self.robot.reset_bin()                # reset the tools
    #             self.parallel_jaw_tool = ParallelJawYuMiTool(self.robot,
    #                                                          self.T_robot_world,
    #                                                          self._parallel_jaw_config,
    #                                                          debug=self.config['debug'])
    #             self.suction_tool = SuctionYuMiTool(self.robot,
    #                                                 self.T_robot_world,
    #                                                 self._suction_config,
    #                                                 debug=self.config['debug'])
    #             self.right_push_tool = PushYuMiTool(self.robot,
    #                                                 self.T_robot_world,
    #                                                 self._right_push_config,
    #                                                 debug=self.config['debug'])
    #             self.left_push_tool = PushYuMiTool(self.robot,
    #                                                self.T_robot_world,
    #                                                self._left_push_config,
    #                                                debug=self.config['debug'])
    #             self.parallel_jaw_tool.open_gripper()
    #             self.suction_tool.open_gripper()                # mark initialized
    #             initialized = True

if __name__ == "__main__": 
    # test_movement()
    test_camera_phoxipy()
    # test_grasp()