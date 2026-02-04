import numpy as np
# from diffusion_policy.env.pusht.pymunk_override import DrawOptions
import gym
from gym import spaces
import click
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
# import zarr
from diffusion_policy.env.pusht.pymunk_override import DrawOptions
import collections
from diffusion_policy.env.pusht.replay_buffer import ReplayBuffer
# import clock

def pymunk_to_shapely(body, shapes):
    geoms = list()
    vertices = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):            
            # print(shape.get_vertices())
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
            vertices += [verts]
            # print(verts)
        # verts = []
        #     for v in shape.get_vertices():
        #         x,y = v.rotated(shape.body.angle) + shape.body.position        
        #         vector = Vec2d(x,y)
        #         verts.append(vector)
        #     verts += [verts[0]]
        #     geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    # print(geoms[0])
    geom = sg.MultiPolygon(geoms)
    return geom, vertices

COLORS = {
    "WHITE":   (255, 255, 255),
    "BLACK":   (0, 0, 0),
    "RED":     (255, 0, 0),
    "GREEN":   (0, 255, 0),
    "BLUE":    (0, 0, 255),
    "YELLOW":  (255, 255, 0),
    "CYAN":    (0, 255, 255),
    "MAGENTA": (255, 0, 255),
    "GRAY":    (128, 128, 128),
    "ORANGE":  (255, 165, 0),
    "PURPLE":  (128, 0, 128),
    "BROWN":   (165, 42, 42)
}


class MultiPushEnvFull(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,            
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            signal_idx = 0):
        # super().__init__()
        # self.render_size = render_size
        # self.num_blocks = num_blocks
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=1, shape=(num_blocks * 2,), dtype=np.float32)
        # self.action_space = gym.spaces.Discrete(num_blocks)
        
        # Initialize the boxes, circles, and the goal position as well as the blinking light color.

        self._seed = None
        self.seed()
        self.window_size = ws = 520   # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        self.red_done = False
        self.goal_poses = [(50,200), (250,200), (450, 200)]
        
        self.signal_circle_poses = [(50,80), (250,80), (450, 80)]
        self.signal_circle_radius = 20
        self.signal_idx = np.random.randint(0,3)
        self.current_goal_pose = np.array([self.goal_poses[self.signal_idx][0], self.goal_poses[self.signal_idx][1], 0])

        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        self.signal_occured = False
        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.box_color_dict = {"box1":COLORS["MAGENTA"], "box2":COLORS["ORANGE"], "box3":COLORS["BROWN"]}
        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )
 
        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None
        self.blink_counter = 0
        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

        self.max_score = 50 * 100
        self.success_threshold = 0.95 
        self.agent = {}
        self.boxes = []
        self.current_box = pymunk.Body()

    def reset(self):
        # self.blocks = np.random.rand(self.num_blocks, 2) * self.render_size
        self._setup()
        self.blink_counter = 0  # Reset blink_counter here so it flashes at the start of each episode
        # self.current_box = self.boxes[0]
        # print(f"The current box is {self.current_box}")
        # return self.blocks.flatten()
        observation = self._get_obs()
        return observation
    
    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position:tuple, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position:tuple, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        # print(f"The shape of the created box is {shape}")
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body
    
    def _get_obs(self):
        img = self.render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }
        # print((obs['image'].shape))
        # print((obs['agent_pos'].shape))
        return obs
    
    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 70))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = 0

        # Add a shape to the body
        shape = pymunk.Poly.create_box(body, (50, 70))  # Create a box shape
        # self.space.add(body, shape)  # Add the body and shape to the space
        return body

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        self.red_done = False
        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        # self.space.add(*walls)
        self.current_box = self.add_box((250, 400), 50,70)
        # self.boxes.append(self.add_box((250, 300), 40,40))
        # self.boxes.append(self.add_box((350, 300), 40,40)) # Add different color names down the line
        self.agent = self.add_circle((250, 350),15)
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        # n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.current_box.position) + [self.current_box.angle]),
            'goal_pose': self.current_goal_pose,
            # 'n_contacts': n_contact_points_per_step
            }
        return info

    def change_circle_color(self, circle_idx, screen):
        pygame.draw.circle(screen,self.box_color_dict[f"box{circle_idx+1}"],
                              (self.signal_circle_poses[circle_idx][0], self.signal_circle_poses[circle_idx][1]),
                               self.signal_circle_radius)

    def render_frame(self,mode):
        flash_color = False
        # print(f"The blink counter is {self.blink_counter}")
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        if self.blink_counter < 80:
            flash_color = True
            

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)


        # Draw the 3 goal poses.
        for i, goal_pose in enumerate(self.goal_poses):
            # print(goal_pose)
            rect = pygame.Rect(goal_pose[0] - 25, goal_pose[1] - 35, 50, 70) #Expects the coordinates of the top left corner
            pygame.draw.rect(canvas, self.box_color_dict[f"box{i+1}"], rect=rect)
            pygame.draw.circle(canvas,COLORS["BLACK"],
                              (self.signal_circle_poses[i][0], self.signal_circle_poses[i][1]),
                               self.signal_circle_radius)
        # if flash_color:
        self.change_circle_color(self.signal_idx, canvas)
        # print("Changing color")

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        self.blink_counter += 1
        return img

    def render(self, mode):
        return self.render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward

        goal_body = self._get_goal_pose_body(self.current_goal_pose)
        # print(goal_body.shapes)
        goal_geom,_ = pymunk_to_shapely(goal_body, self.current_box.shapes)
        # print(self.current_box.shapes)
        block_geom,_ = pymunk_to_shapely(self.current_box, self.current_box.shapes)
        # print(block_geom)
        # goal_red_body = self._get_goal_pose_body(self.goal_red_pose)
        # goal_red_geom = pymunk_to_shapely(goal_red_body, self.block.shapes)
        # # block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        # # print(block_geom.area)
        # if not self.red_done:
        # intersection_red_area = goal_red_geom.intersection(block_geom).area
        #     # print(intersection_red_area)
        #     goal_red_area = goal_red_geom.area
        #     coverage_red = intersection_red_area / goal_red_area
        #     print(f"The coverage of the blue area is {coverage_red}")# print(coverage_red)
        #     reward_red  = np.clip(coverage_red / self.success_threshold, 0, 1)
        #     self.red_done = True if (reward_red == 1) else False
        
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        # print(intersection_area, goal_area)

        # print(f"The current coverage is {coverage}")    
        reward = np.clip(coverage / self.success_threshold, 0, 1)

        # done =  (self.red_done) and (coverage > self.success_threshold)
        # print(done)
        observation = self._get_obs()
        # info = self._get_info()
        info = self._get_info()
        if reward == 1:
            assert observation is not None, "env._get_obs() returned None"
            assert info is not None, "env._get_info() returned None"
            return observation, reward, True, info
        # else:
        #     return observation, reward, False, info
        # if self.red_done:
        #     print("RED is done onto greem")
        #     # reward = reward_red 
        # else:
        #     reward = 0
        # reward = 1
        # done = False # !!! CHANGE THIS!!!!
        assert observation is not None, "env._get_obs() returned None"
        assert info is not None, "env._get_info() returned None"
        return observation, reward, False, info
    

@click.command()
@click.option('-o', '--output', required=True)
def main(output):
    # push_env = MultiPushEnv(signal_idx=np.random.randint(0,3))
    # agent = push_env.teleop_agent()
    # clock = pygame.time.Clock()
    plan_idx = 0
    while True:
        replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
        print(f"Index count is {replay_buffer.n_episodes}")
        seed = replay_buffer.n_episodes
        push_env = MultiPushEnv(signal_idx=np.random.randint(0,3))
        agent = push_env.teleop_agent()
        clock = pygame.time.Clock()
        push_env.reset()
        # push_env.signal_idx = np.random.randint(0,3)
        push_env.render(mode="human")
        episode = list()
        push_env.signal_occured = False
        # record in seed order, starting with 0
        # seed = replay_buffer.n_episodes
        # print(f'starting seed {seed}')

        # set seed for env
        # push_env.seed(seed)
        
        # reset push_ev and get observations (including info and render for recording)
        obs = push_env.reset()
        info = push_env._get_info()
        img = push_env.render(mode='human')
        
        # loop state
        retry = False
        pause = False
        done = False
        
        pygame.display.set_caption(f'plan_idx:{plan_idx}')
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                break
            if pause:
                continue
            
            # get action from mouse
            # None if mouse is not close to the agent
            obs = []
            
            # if push_env.signal_occured == False:
            #     for i in range(80):
                    
            #         img = push_env.render_frame(mode='human')
            #         act = agent.act(obs)
            #         obs, reward, done, info = push_env.step(act)
            #         state = np.concatenate([info['pos_agent'], info['block_pose']])
            #         data = {
            #         'img': img,
            #         'state': np.float32(state),
            #         # 'keypoint': np.float32(keypoint),
            #         'action': np.float32(act),
            #         # 'n_contacts': np.float32([info['n_contacts']])
            #         }
            #         if act is not None:
            #             episode.append(data)
            #         clock.tick(10)
            #         push_env.signal_occured = True
            # regulate control frequency
            clock.tick(10)
            # if not act is None:
            #     # teleop started
            #     # state dim 2+3
                
            #     # discard unused information such as visibility mask and agent pos
            #     # for compatibility
            #     keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:9]
            act = agent.act(obs)
            obs, reward, done, info = push_env.step(act)
            # info = push_env._get_info()
            state = np.concatenate([info['pos_agent'], info['block_pose']])
            img = push_env.render_frame(mode='human')
            # print(f"The current action is {img}")

            data = {
                'img': img,
                'state': np.float32(state),
                # 'keypoint': np.float32(keypoint),
                'action': np.float32(act),
                # 'n_contacts': np.float32([info['n_contacts']])
            }
            if act is not None:
                episode.append(data)
                
        
                
            # step push_env and render
            
            # done = False
            # print(f"The current observation is {obs}")
            
            tick = False
        print(len(episode))
        if not retry:
                    # save episode buffer to replay buffer (on disk)
                    data_dict = dict()
                    for key in episode[0].keys():
                        print(key)
                        data_dict[key] = np.stack(
                            [x[key] for x in episode])
                        print('here')
                    replay_buffer.add_episode(data_dict, compressors='disk')
                    print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')
        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            plan_idx += 1
            print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')

if __name__=="__main__":
    main()