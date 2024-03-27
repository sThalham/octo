# minimum working example to load a single OXE dataset
from octo.data.oxe import make_oxe_dataset_kwargs
from octo.data.dataset import make_single_dataset
from PIL import Image
import numpy as np

dataset_kwargs = make_oxe_dataset_kwargs(
    # see octo/data/oxe/oxe_dataset_configs.py for available datasets
    # (this is a very small one for faster loading)
    "austin_buds_dataset_converted_externally_to_rlds",
    # can be local or on cloud storage (anything supported by TFDS)
    # "/path/to/base/oxe/directory",
    "gs://gresearch/robotics",
)
dataset = make_single_dataset(dataset_kwargs, train=True) # load the train split
iterator = dataset.iterator()

# make_single_dataset yields entire trajectories

for ita, dummy in enumerate(iterator):

    traj = next(iterator)
    top_keys = traj.keys()
    observation = traj["observation"]
    task = traj["task"]
    action = traj["action"]
    absolute_action_mask = traj["absolute_action_mask"]

    print("vidya stream: ", traj["observation"]["image_primary"].shape)
    #for actit in range(task["language_instruction"].shape[0]):
    print(task)
    print("language instruction: ", task["language_instruction"])#[actit])
    print("What is proprio: ", observation["proprio"].shape)
    print("What's in action?: ", action.shape)
    print("What's the mask doing?: ", absolute_action_mask.shape)

traj = next(iterator)
images = traj["observation"]["image_primary"]
# should be: (traj_len, window_size, height, width, channels)
# (window_size defaults to 1)
print(images.shape)
Image.fromarray(np.concatenate(images.squeeze()[-5:], axis=1))