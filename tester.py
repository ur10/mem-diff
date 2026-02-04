from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.env.pusht.multi_push import MultiPushEnv
env = MultiStepWrapper(
    VideoRecordingWrapper(MultiPushEnv(render_size=96),
                    video_recoder=VideoRecorder.create_h264(
                        fps=10,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=22,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,),
    n_obs_steps=8,
    n_action_steps=8,
    max_episode_steps=200
)
obs = env.reset()
action = env.action_space.sample()
obs, rew, done, info = env.step(action)