import time
from pymycobot import MyCobot280
from pymycobot.genre import Angle, Coord
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from functools import partial
import imageio
import os
import cv2
from absl import app, flags, logging
import click

from transformers import AutoModel
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper
from octo.utils.train_callbacks import supply_rng

# Path to save video
VIDEO_SAVE_PATH = "./mycobot_videos"
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "base", "Pretrained model. one of [\"small\", \"base\"]") # 4 for octo_base

# custom to bridge_data_robot
flags.DEFINE_integer("baud", 115200, "Baud rate of MyCobot280")
flags.DEFINE_string("port_robot", "/dev/ttyUSB0", "Port of the robot")
flags.DEFINE_string("port_camera", "/dev/video0", "Port of the camera")
flags.DEFINE_spaceseplist("goal_eep", [-4.13, 20.91, -133.06, 19.86, 17.84, -49.13], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [-4.74, 18.54, -11.25, -77.6, 11.33, -63.45], "Initial position")
flags.DEFINE_bool("blocking", False, "Use the blocking controller") # is it needed for the MyCobot280?


#flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", VIDEO_SAVE_PATH, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer(
    "action_horizon", 4, "Length of action sequence to execute/ensemble"
)


# show image flag
flags.DEFINE_bool("show_image", True, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
Be sure to use a step duration of 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################

def preprocess_image(img):
    height, width, _ = img.shape
    img = img[:, int((width*0.5)-(height*0.5)):int((width*0.5)+(height*0.5)), :]
    return cv2.resize(img, (256,256))

def main(_):
    # MyCobot initialization
    mycobot = MyCobot280(FLAGS.port_robot, FLAGS.baud)
    cam_cap = cv2.VideoCapture(FLAGS.port_camera)
    if not cam_cap.isOpened():
        print("Cannot open camera")
        exit()

    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        mycobot.send_angles(initial_eep, 50)
        time.sleep(3)
    else:
        print("Please set initial pose for robot")
        exit()

    if not FLAGS.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE

    # load models
    if FLAGS.model == "small":
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
        im_size = 128
    elif FLAGS.model == "base":
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
        im_size = 256
    else:
        print("Flag for model invalid; valid flags [\"small\", \"base\"]")

    def sample_actions(
            pretrained_model: OctoModel,
            observations,
            tasks,
            rng,
    ):
        # add batch dim to observations
        #observations = jax.tree_map(lambda x: x[None], observations)
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations=observations,
            tasks=tasks,
            rng=rng,
            unnormalization_statistics=pretrained_model.dataset_statistics[
                "bridge_dataset"
            ]["action"],
        )
        # remove batch dim
        return actions[0]

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            #argmax=False, # no idea is True or False for deterministic
            #temperature=1.0, # default Octo value
        )
    )

    goal_image = jnp.zeros((im_size, im_size, 3), dtype=np.uint8)
    goal_instruction = ""

    # goal sampling loop
    while True:
        modality = click.prompt(
            "Language or goal image?", type=click.Choice(["l", "g"])
        )

        if modality == "g":
            if click.confirm("Take a new goal?", default=True):
                assert isinstance(FLAGS.goal_eep, list)
                _eep = [float(e) for e in FLAGS.goal_eep]

                goal_eep = FLAGS.goal_eep
                #goal_eep = state_to_eep(_eep, 0)
                mycobot.set_gripper_state(1, speed=50)  # open gripper

                move_status = None
                #while move_status != WidowXStatus.SUCCESS:
                #    move_status = widowx_client.move(goal_eep, duration=1.5)
                mycobot.send_angles(goal_eep, 50)
                time.sleep(3)

                input("Press [Enter] when ready for taking the goal image. ")
                #obs = wait_for_obs(widowx_client)
                #obs = convert_obs(obs, FLAGS.im_size)
                ret, frame = cam_cap.read()
                frame = preprocess_image(frame)
                obs = {"image_primary": frame, "timestep_pad_mask": np.array([[True]])}
                goal = jax.tree_map(lambda x: x[None], obs)

            mycobot.send_angles(initial_eep, 50)
            time.sleep(3)

            # Format task for the model
            task = model.create_tasks(goals=goal)
            # For logging purposes
            goal_image = goal["image_primary"][0]
            goal_instruction = ""

        elif modality == "l":
            print("Current instruction: ", goal_instruction)
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?")

            print("text instruction: ", text)
            # Format task for the model
            task = model.create_tasks(texts=[text])
            # For logging purposes
            goal_instruction = text
            goal_image = jnp.zeros_like(goal_image)
        else:
            raise NotImplementedError()

        input("Press [Enter] to start.")

        # reset env
        #obs, _ = env.reset()
        time.sleep(2.0)

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()

                ret, frame = cam_cap.read()
                frame = preprocess_image(frame)
                frame= frame[np.newaxis, ...]
                timestep_pad = np.array([True, True])
                #timestep_pad = timestep_pad[np.newaxis, ...]
                print(timestep_pad.shape)

                # save images
                images.append(frame)
                goals.append(goal_image)

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # get action
                forward_pass_time = time.time()

                # manual manipulation of observation
                if not images:
                    frame = np.concatenate([frame, frame], axis=0)
                else:
                    frame = np.concatenate([images[-1], frame], axis=0)
                print("shape frame: ", frame.shape)

                obs = {"image_primary": frame, "timestep_pad_mask": timestep_pad}

                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time: ", time.time() - forward_pass_time)

                # perform environment step
                start_time = time.time()

                print(action[0][:6])
                #obs, _, _, truncated, _ = env.step(action)
                mycobot.send_angles(list(action[0][:6]), 50)
                print("step time: ", time.time() - start_time)

                t += 1

                #if truncated:
                #    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)


