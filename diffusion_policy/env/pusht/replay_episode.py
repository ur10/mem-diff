from replay_buffer import ReplayBuffer
import click
import cv2
import numpy as np


def replay_episode(replay_buffer, render_size=(256,256)):
    """
    Replay and record episodes in PushTKeypointsEnv with teleop agent.
    
    Controls:
    """
    a = 1
    episode = replay_buffer.pop_episode()
    imgs = episode['img']
    # Ensure imgs is a numpy array of shape (N, H, W, C)
    imgs = np.array(imgs)
    height, width = render_size
    # Resize images if needed
    resized_imgs = [cv2.resize(img, (width, height)) for img in imgs]

    # Define video writer
    out_path = "/home/ur10/rollout.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20  # or set to your episode's frame rate
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for img in resized_imgs[:-110]:
        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # If grayscale, convert to BGR
        if len(img.shape) == 2 or img.shape[2] == 1:
            print('grayscale image detected, converting to BGR')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # print(img)
        out.write(img)
    for i in range(80):
        out.write(resized_imgs[-110])
    out.release()



@click.command()
@click.option('-o', '--output', required=True)
def main(output):
    """
    Replay and record episodes in PushTKeypointsEnv with teleop agent.
    
    Usage: python replay_episode.py -o data/pusht_demo.zarr
    """
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
    replay_episode(replay_buffer)

if __name__ == "__main__":
    main()