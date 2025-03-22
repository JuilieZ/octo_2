import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .rm75 import RM75


@register_agent()
class RM75WristCam(RM75):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "rm75_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/rm75/urdf/rm75_wristcam.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                # [0.707,0,-0.707,0]
                # ground truth [0.5,-0.5,-0.5,-0.5]
                pose=sapien.Pose(p=[0.01,0,0.01], q=[1,0,0,0]),
                width=640,
                height=480,
                fov=np.pi/180*60,
                near=0.01,
                far=100,
                mount=self.robot.links_map["Link8"],
                # "sensor_link"
            )
        ]