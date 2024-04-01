# minimum working example to load a single OXE dataset
from octo.data.oxe import make_oxe_dataset_kwargs, make_oxe_dataset_kwargs_and_weights
from octo.data.dataset import make_single_dataset, make_interleaved_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#dataset_kwargs = make_oxe_dataset_kwargs(
    # see octo/data/oxe/oxe_dataset_configs.py for available datasets
    # (this is a very small one for faster loading)
#    "austin_buds_dataset_converted_externally_to_rlds",
    # can be local or on cloud storage (anything supported by TFDS)
    # "/path/to/base/oxe/directory",
#    "gs://gresearch/robotics",
#)

dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
    # you can pass your own list of dataset names and sample weights here, but we've
    # also provided a few named mixes for convenience. The Octo model was trained
    # using the "oxe_magic_soup" mix.
    "rtx",
    # can be local or on cloud storage (anything supported by TFDS)
    "gs://gresearch/robotics",
    # let's get a wrist camera!
    load_camera_views=("primary", "wrist"),
)

#dataset = make_single_dataset(dataset_kwargs, train=True) # load the train split
#iterator = dataset.iterator()

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 8

dataset = make_interleaved_dataset(
    dataset_kwargs_list,
    sample_weights,
    train=True,
    # unlike our manual shuffling above, `make_interleaved_dataset` will shuffle
    # the JPEG-encoded images, so you should be able to fit a much larger buffer size
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    # see `octo.data.dataset.apply_trajectory_transforms` for full documentation
    # of these configuration options
    traj_transform_kwargs=dict(
        goal_relabeling_strategy="uniform",  # let's get some goal images
        window_size=2,  # let's get some history
        future_action_window_size=3,  # let's get some future actions for action chunking
        subsample_length=100,  # subsampling long trajectories improves shuffling a lot
    ),
    # see `octo.data.dataset.apply_frame_transforms` for full documentation
    # of these configuration options
    frame_transform_kwargs=dict(
        # let's apply some basic image augmentations -- see `dlimp.transforms.augment_image`
        # for full documentation of these configuration options
        image_augment_kwargs=dict(
            augment_order=["random_resized_crop", "random_brightness"],
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.1],
        ),
        # provided a `resize_size` is highly recommended for a mixed dataset, otherwise
        # datasets with different resolutions will cause errors
        resize_size=dict(
            primary=(256, 256),
            wrist=(128, 128),
        ),
        # If parallelism options are not provided, they will default to tf.Data.AUTOTUNE.
        # However, we would highly recommend setting them manually if you run into issues
        # with memory or dataloading speed. Frame transforms are usually the speed
        # bottleneck (due to image decoding, augmentation, and resizing), so you can set
        # this to a very high value if you have a lot of CPU cores. Keep in mind that more
        # parallel calls also use more memory, though.
        num_parallel_calls=64,
    ),
    # Same spiel as above about performance, although trajectory transforms and data reading
    # are usually not the speed bottleneck. One reason to manually set these is if you want
    # to reduce memory usage (since autotune may spawn way more threads than necessary).
    traj_transform_threads=16,
    traj_read_threads=16,
)

# Another performance knob to tune is the number of batches to prefetch -- again,
# the default of tf.data.AUTOTUNE can sometimes use more memory than necessary.

for ita, dummy in enumerate(iterator):

    print(ita)

    traj = next(iterator)
    top_keys = traj.keys()
    observation = traj["observation"]
    task = traj["task"]
    action = traj["action"]
    absolute_action_mask = traj["absolute_action_mask"]

    print("vidya stream: ", traj["observation"]["image_primary"].shape)
    vis_observation = traj["observation"]["image_primary"]
    #for actit in range(task["language_instruction"].shape[0]):
    print(task.keys())
    #print(action.keys())
    #print(absolute_action_mask.keys())
    print("language instruction: ", np.unique(task["language_instruction"]))#[actit])
    print("language instruction mask: ", np.unique(task["pad_mask_dict"]["language_instruction"]))
    #print("What is proprio: ", observation["proprio"].shape)
    #print("What's in action?: ", action.shape)
    #print("What's the mask doing?: ", absolute_action_mask.shape)

    imgs = vis_observation[0::10, ...].squeeze()
    print('imgs: ', imgs.shape)
    r1 = np.concatenate(imgs[:10, ...], axis=0)
    r2 = np.concatenate(imgs[10:20, ...], axis=0)
    r3 = np.concatenate(imgs[20:30, ...], axis=0)
    r4 = np.concatenate(imgs[30:40, ...], axis=0)
    r5 = np.concatenate(imgs[40:50, ...], axis=0)
    r6 = np.concatenate(imgs[50:60, ...], axis=0)
    print('r1: ', r1.shape)
    #print(imgs.shape)
    #print(vis_observation.squeeze().shape)
    im = Image.fromarray(np.transpose(np.concatenate([r1, r2, r3, r4, r5, r6], axis=1), (1, 0, 2)))
    plt.imshow(im)
    plt.show()

    #im.show()
#traj = next(iterator)
#images = traj["observation"]["image_primary"]
# should be: (traj_len, window_size, height, width, channels)
# (window_size defaults to 1)
#print(images.shape)
#Image.fromarray(np.concatenate(images.squeeze()[-5:], axis=1))