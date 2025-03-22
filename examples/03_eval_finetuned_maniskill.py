"""
This script demonstrates how to load and rollout a finetuned Octo model.
We use the Octo model finetuned on ALOHA sim data from the examples/02_finetune_new_observation_action.py script.

For installing the ALOHA sim environment, clone: https://github.com/tonyzhaozh/act
Then run:
pip3 install opencv-python modern_robotics pyrealsense2 h5py_cache pyquaternion pyyaml rospkg pexpect mujoco==2.3.3 dm_control==1.0.9 einops packaging h5py

Finally, modify the `sys.path.append` statement below to add the ACT repo to your path.
If you are running this on a head-less server, start a virtual display:
    Xvfb :1 -screen 0 1024x768x16 &
    export DISPLAY=:1

To run this script, run:
    cd examples
    python3 03_eval_finetuned_maniskill.py --finetuned_path=/home/server/octo/checkpoint/ckpt_06
"""
from functools import partial
import sys

from absl import app, flags, logging
import gymnasium as gym
import jax
import numpy as np
import wandb

# sys.path.append("/home/server/octo/examples/mani_skill")

# keep this to register ALOHA sim env
# from envs.aloha_sim_env import AlohaGymEnv  # noqa
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng
from mani_skill.agents.controllers.base_controller import CombinedController

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)


def main(_):
    # setup wandb for logging
    wandb.init(name="eval_maniskill_pickcube_pd_ee_target_pose_without_norm", project="octo")

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path)
    # model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
   

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_primary": ...
    #     }
    #   }
    ##################################################################################################################
    env = gym.make("PickCube-v1")
    # env = gym.make("PullCube-v1")
    obs, info = env.reset()
    # print(env.shader)
    # wrap env to normalize proprio
    # env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)
    obs, info = env.reset()
    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    controller: CombinedController = env.agent.controller
    controller.controllers["arm"].config.use_target = True
    controller.controllers["arm"].config.use_delta = False
    controller.controllers["arm"].config.normalize_action = False
    # arm_pd_ee_delta_pose.frame = "root_translation:root_aligned_body_rotation"
    # print(controller.controllers["arm"].config)
    # breakpoint()
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    # running rollouts
    for id in range(5):
        obs, info = env.reset()
        # print(obs['proprio'])
        # breakpoint()

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        
        task = model.create_tasks(texts=language_instruction)
        # run rollout for 400 steps
        images = [obs["image_wrist"][0]]
        episode_return = 0.0
        while len(images) < 400:
            import torch
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0]
            # print(actions)
            # breakpoint()
            # import jax.numpy as jnp
            # new_actions = jnp.zeros((50, 8))
            # from scipy.spatial.transform import Rotation
            # # Create a rotation object from Euler angles specifying axes of rotation
            # new_actions = jnp.zeros((50, 8))  
            # for i in range(50):
            #     new_actions = new_actions.at[i].set(jnp.hstack([actions[i],actions[i,-1]]))
            # print("action>>>",new_actions.shape)
            # raise KeyboardInterrupt
            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            # print(obs['proprio'])
            # breakpoint()
            images.extend([o["image_wrist"][0] for o in info["observations"]])
            episode_return += reward
            if done or trunc:
                if done:
                    print("Oops")
                break
        print(f"{id}:Episode return: {episode_return}")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log(
            {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        )


if __name__ == "__main__":
    app.run(main)
