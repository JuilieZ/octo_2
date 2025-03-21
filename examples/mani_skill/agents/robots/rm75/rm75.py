from copy import deepcopy

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig

@register_agent()
class RM75(BaseAgent):
    uid = "rm75"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/rm75/urdf/rm75.urdf"
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0,
                    0,
                    0,
                    np.pi*1/2,
                    0,
                    np.pi*1/2,
                    0,
                    # 0. ,
                    # 0.90381875 ,
                    # -1.26545097, 
                    # 0.31368803 ,
                    # -1.14997999 ,
                    # -1.04728482,
                    # -0.6493672,
                    0.8,
                    0.4,
                    0.8,
                    0.4,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    gripper_joint_names_1 = [
        "eg2_joint1",
        "eg2_joint3",
         ]
    gripper_joint_names_2 = [
        "eg2_joint2",
        "eg2_joint4",
         ]
    ee_link_name = "eg2_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit =100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_stiffness_2 = 1e4
    gripper_damping_2 = 1e0
    gripper_force_limit = 1e10
    # gripper_force_limit = 1e1

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos_1 = PDJointPosMimicControllerConfig(
            self.gripper_joint_names_1,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.8,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )
        gripper_pd_joint_pos_2 = PDJointPosMimicControllerConfig(
            self.gripper_joint_names_2,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.8,
            stiffness=self.gripper_stiffness_2,
            damping=self.gripper_damping_2,
            force_limit=self.gripper_force_limit,
        )
        # gripper_pd_joint_pos = list([gripper_pd_joint_pos_1,gripper_pd_joint_pos_2])
        # gripper_pd_joint_pos = gripper_pd_joint_pos_1

        # controller_configs = dict(
        #     pd_joint_delta_pos=dict(
        #         arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
        #     ),
        #     pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
        #     pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
        #     pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos ),
        #     pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
        #     # TODO(jigu): how to add boundaries for the following controllers
        #     pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos,gripper=gripper_pd_joint_pos),
        #     pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos),
        #     pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos),
        #     # Caution to use the following controllers
        #     pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
        #     pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos),
        #     pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos  ),
        # )
        controller_configs = dict(pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos,gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos,gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel, gripper_1=gripper_pd_joint_pos_1,gripper_2=gripper_pd_joint_pos_2),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.finger11_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "eg2_Link1"
        )
        self.finger12_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "eg2_Link2"
        )
        self.finger21_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "eg2_Link3"
        )
        self.finger22_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "eg2_Link4"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger12_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger22_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger12_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger22_link.pose.to_transformation_matrix()[..., :3, 1]
        # print(ldirection[0,0])
        ldirection[0,0] ,ldirection[0,1] ,ldirection[0,2] = -ldirection[0,0] ,-ldirection[0,1] ,ldirection[0,2] 
        rdirection[0,0] ,rdirection[0,1] ,rdirection[0,2] = -rdirection[0,0] ,-rdirection[0,1] ,rdirection[0,2] 


        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        
        # print("===========force_vec",l_contact_forces,r_contact_forces)
        # print("===========finger_direction",ldirection,rdirection)
        # print("===========angle",langle,rangle)

        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-4]
        # print(qvel)
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (eg2_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    # @property
    # def _sensor_configs(self):
    #     return [
    #         CameraConfig(
    #             uid="hand_camera",
    #             pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             mount=self.robot.links_map["camera_link"],
    #         )
    #     ]
