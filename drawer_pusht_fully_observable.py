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
import math
# import zarr
from diffusion_policy.env.pusht.pymunk_override import DrawOptions
import collections
from diffusion_policy.env.pusht.replay_buffer import ReplayBuffer
# import clock

def pymunk_to_shapely(body, shapes, circle_segments=32):
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
        elif isinstance(shape, pymunk.shapes.Circle):
            # Get circle center in world coordinates
            center = body.local_to_world(shape.offset)
            radius = shape.radius
            
            # Generate polygon vertices to approximate the circle
            verts = []
            for i in range(circle_segments):
                angle = 2 * math.pi * i / circle_segments
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                verts.append((x, y))
            verts.append(verts[0])  # Close the polygon
            
            geoms.append(sg.Polygon(verts))
            vertices += [verts]
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

COLOR_PALLETE = [{"1":(128, 128, 128), "2": (255, 165, 0) }, {"2":(128, 128, 128), "1": (255, 165, 0)}]
class DrawerPushtFull(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)


    def __init__(self,            
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            signal_pos_idx = np.random.randint(0,3), 
            signal_color_idx = np.random.randint(0,3)):
        # super().__init__()
        # self.render_size = render_size
        # self.num_blocks = num_blocks
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=1, shape=(num_blocks * 2,), dtype=np.float32)
        # self.action_space = gym.spaces.Discrete(num_blocks)
        
        # Initialize the boxes, circles, and the goal position as well as the blinking light color.
        # signal_pos_idx = signal_pos_idx % 2
        # signal_color_idx = signal_color_idx % 2
        self.pos_choice_array = [[1,0,1],[0,1,0],[1,1,0],[0,0,1],[1,0,0],[0,1,1],[1,1,1],[0,0,0]]
        self.color_choice_array = [[1,0,1],[0,1,0],[1,1,0],[0,0,1],[1,0,0],[0,1,1],[1,1,1],[0,0,0]]
        signal_pos_idx  =  self.pos_choice_array[signal_pos_idx][signal_pos_idx]
        signal_color_idx  =  self.color_choice_array[signal_color_idx][signal_color_idx]
        print(f"The signal position index is {signal_pos_idx} and the signal color index is {signal_color_idx}")
        self._seed = None
        self.seed()
        self.window_size = ws = 520   # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        self.red_done = False
        self.goal_poses = [(150,100), (150,350)]
        self.block_position = [(150, 120), (150, 400)]
        self.signal_circle_poses = [(450,80), (450,80), (450, 80)]
        self.signal_circle_radius = 20
        self.signal_pos_idx = signal_pos_idx
        self.signal_color_idx = signal_color_idx
        self.current_goal_pose = np.array([450,80, 0])
        self.chosen_palet = COLOR_PALLETE[signal_color_idx]
        self.box_contacted = [False, False]
        self.box_contacted_count = [0,0]
        self.first_step = True
        print(self.chosen_palet)
        self.intermediate_circle_covered = False
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy
        self.correct_block_taken = False

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
        self.box_color_dict = {"box1":COLORS["MAGENTA"], "box2":COLORS["CYAN"], "box3":COLORS["BROWN"]}
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
        self.intermediate_circle= pymunk.Body()
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

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
        self.intermediate_count = 0
        self.max_score = 50 * 100
        self.success_threshold = 0.95 
        self.agent = {}
        self.boxes = []
        self.current_box = pymunk.Body()

    def reset(self):
        # self.blocks = np.random.rand(self.num_blocks, 2) * self.render_size
        self._setup()
        observation = self._get_obs()
        return observation
        # self.current_box = self.boxes[0]
        # print(f"The current box is {self.current_box}")
        # return self.blocks.flatten()
    
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

    def add_box(self, position:tuple, height, width, color):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color(color)
        shape.outline_color = None  # Remove boundary/outline 
        self.space.add(body, shape)
        return body
        # mass = 1
        # inertia = pymunk.moment_for_box(mass, (height, width))
        # body = pymunk.Body(mass, inertia)
        # body.position = position
        # shape = pymunk.Poly.create_box(body, (height, width))
        # # print(f"The shape of the created box is {shape}")
        
        # shape.color = pygame.Color(self.box_color_dict[f"box{self.signal_pos_idx}"])
        # self.space.add(body, shape)
        # return body
    
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
        choices = [0,1]
        draw = np.random.choice(choices, size=2, replace=False)
        colors = ["GRAY" ,"ORANGE"]
        self.boxes = [
            self.add_box(self.block_position[draw[0]], 50, 70, COLORS["GRAY"]),
            self.add_box(self.block_position[draw[1]], 50, 70, COLORS["ORANGE"])
        ]
        self.current_box = self.boxes[self.signal_pos_idx]
        print(self.current_box)
        self.current_color = "GRAY" if self.signal_pos_idx == 0 else "ORANGE"
       
        # self.current_box = self.add_box(self.block_position[1], 50,70, COLORS["CYAN"])

        # self.boxes.append(self.add_box((250, 300), 40,40))
        # self.boxes.append(self.add_box((350, 300), 40,40)) # Add different color names down the line
        self.agent = self.add_circle((450, 200),15)
        self.intermediate_circle = self.add_circle((450, 200),20)
    
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

    def check_change_color(self):
        # for i, goal_pose in enumerate(self.goal_poses):
        #     if (abs(self.agent.position[0] - goal_pose[0]) < 60.0 and abs(self.agent.position[1] - goal_pose[1]) < 60.0):
        #         shape = self.space.shapes[i]
        #         shape.color = pygame.Color(self.chosen_palet[f"{i+1}"])
        #         self.intermediate_circle_covered = False
        #     else:
        #         shape = self.space.shapes[i]
        #         # print((shape.body.position - self.agent.position).length)
        #         if abs((shape.body.position - self.agent.position).length) > 60 and (self.box_contacted[i] == False):
                                    
        #             shape.color = pygame.Color(self.box_color_dict[f"box{i+1}"])
        #             # print(f"The box color of box {i+1} is {self.box_color_dict[f'box{i+1}']}")
        #         else:
        #             self.box_contacted[i] = True
        #             # shape.color = pygame.Color(self.chosen_palet[f"{i+1}"])
        #             self.correct_block_taken = True
        #             self.intermediate_circle_covered = True
            
        #     shape = self.space.shapes[i]
        #     if abs((shape.body.position - self.agent.position).length) < 60:
        #             self.box_contacted_count[i] += 1
        #     if self.box_contacted_count[i] > 100:
        #         self.box_contacted[i] = True
        #         shape = self.space.shapes[i]
        #         shape.color = pygame.Color(self.chosen_palet[f"{i+1}"])

        #         print(f"Box {i+1} contacted count {self.box_contacted_count[i]}")

        for i, goal_pose in enumerate(self.goal_poses):
            shape = self.space.shapes[i]
            
            # Calculate distance from agent to shape
            distance_to_shape = abs((shape.body.position - self.agent.position).length)
            
            # Update contact counter when agent is close
            if distance_to_shape < 60:
                self.box_contacted_count[i] += 1
            
            # Mark as permanently contacted after sustained proximity (100+ frames)
            if self.box_contacted_count[i] > 60:
                if not self.box_contacted[i]:  # First time reaching threshold
                    # print(f"Box {i+1} permanently contacted - count: {self.box_contacted_count[i]}")
                    self.box_contacted[i] = True
            
            # Update color based on proximity and permanent contact state
            if self.box_contacted[i]:
                # Permanently contacted - always use chosen color
                shape.color = pygame.Color(self.chosen_palet[f"{i+1}"])
                self.correct_block_taken = True
                self.intermediate_circle_covered = True
                
            elif distance_to_shape < 60:
                self.first_step = False
                # Agent is near but not permanently contacted - temporarily change color
                shape.color = pygame.Color(self.chosen_palet[f"{i+1}"])
                # self.intermediate_circle_covered = True
                
            else:
                # Agent is far and not permanently contacted - use default color
                shape.color = pygame.Color(self.box_color_dict[f"box{i+1}"])
                # self.intermediate_circle_covered = False

    def render(self, mode):
        return self.render_frame(mode)
    
    def render_frame(self,mode, flash_color=False):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)
        rect = pygame.Rect(self.signal_circle_poses[0][0] - 25, self.signal_circle_poses[0][1] - 35, 50, 70) #Expects the coordinates of the top left corner
        pygame.draw.rect(canvas, self.current_color, rect=rect)
        # pygame.draw.circle(canvas,self.chosen_palet[f"{self.signal_pos_idx+1}"],
        #                       (self.signal_circle_poses[0][0], self.signal_circle_poses[0][1]),
        #                        self.signal_circle_radius)
        for i, goal_pose in enumerate(self.goal_poses):

            s = pygame.Surface((150, 200), pygame.SRCALPHA)  # Create a surface with alpha channel
            s.fill((*self.box_color_dict[f"box{i+1}"], 250))  # Add alpha value (128 for 50% transparency)
            
            canvas.blit(s, (goal_pose[0] - 75, goal_pose[1] - 100))

        # self.check_change_color()
       
        

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
        return img

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
        reward = 0.0
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

        intermidiate_circle_geom ,_ = pymunk_to_shapely(self.intermediate_circle, self.intermediate_circle.shapes)
        agent_geom ,_ = pymunk_to_shapely(self.agent, self.agent.shapes)
    
        # if not self.intermediate_circle_covered and self.correct_block_taken == False and not self.first_step:
        #     intersection_intermediate_area = agent_geom.intersection(intermidiate_circle_geom).area
        #     intermediate_area = intermidiate_circle_geom.area
        #     coverage_intermediate = intersection_intermediate_area / agent_geom.area
        #     # print(f"The coverage of the intermediate circle is {coverage_intermediate}")# print(coverage_red)
            
        #     if coverage_intermediate > 0.75:
        #         self.intermediate_count +=1
        #         if self.intermediate_count > 30:
        #             self.intermediate_circle_covered = True 
        #             # print("Intermediate circle covered")
        #             reward = 0.5
        #     # else:
        #     #     self.intermediate_circle_covered = False
        #     #     reward = 0.0
        # else:
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        # print(intersection_area, goal_area)

        print(f"The current coverage is {coverage}")    
        reward = np.clip(coverage / self.success_threshold, 0, 1)

        # done =  (self.red_done) and (coverage > self.success_threshold)
        # print(done)
        observation = self._get_obs()
        # info = self._get_info()
        info = self._get_info()
        if reward >= 1:
             return observation, reward, True, info
        else:
            return observation, reward, False, info

        
@click.command()
@click.option('-o', '--output', required=True)
def main(output):
    # push_env = MultiPushEnv(signal_pos_idx=np.random.randint(0,3))
    # agent = push_env.teleop_agent()
    # clock = pygame.time.Clock()
    plan_idx = 0
    while True:
        replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
        seed = replay_buffer.n_episodes
        push_env = DrawerPusht(signal_pos_idx=np.random.randint(0,3), signal_color_idx=np.random.randint(0,3))
        agent = push_env.teleop_agent()
        clock = pygame.time.Clock()
        push_env.reset()
        # push_env.signal_pos_idx = np.random.randint(0,3)
        push_env.render_frame(mode="human")
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
        img = push_env.render_frame(mode='human')
        
        # loop state
        retry = False
        pause = False
        done = False
        # plan_idx = 0
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
                    
            #         img = push_env.render_frame(mode='human', flash_color=((i%4)==0))
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
                    plan_idx += 1
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
        # if not retry:
        #     # save episode buffer to replay buffer (on disk)
        #     data_dict = dict()
        #     for key in episode[0].keys():
        #         data_dict[key] = np.stack(
        #             [x[key] for x in episode])
        #     replay_buffer.add_episode(data_dict, compressors='disk')
        #     print(f'saved seed {seed}')
        # else:
        #     print(f'retry seed {seed}')
if __name__=="__main__":
    main()
'''
Certifiable estimation has revolutionized geometric perception in robotics by providing mathematical guarantees of global optimality for state estimation problems.
Unlike traditional optimization methods that can become trapped in local minima—producing catastrophically incorrect solutions with no way to detect failure—certifiable methods based on semidefinite relaxations can verify whether a recovered estimate is globally optimal or bound how far it deviates from optimality. For safety-critical applications like autonomous drones, aerospace navigation, and mission-critical robotics, this distinction is not academic: the difference between a globally optimal pose estimate and a
local minimum can mean the difference between mission success and catastrophic failure. Methods like SE-Sync have demonstrated that such guarantees are practically achievable for problems involving cameras, LiDAR, and range sensors, handling real-world noise levels while outperforming conventional approaches.
However, a critical limitation exists: current certifiable frameworks are restricted to position-based measurements.

They cannot incorporate velocity and acceleration data from inertial sensors—IMUs, gyroscopes, Doppler radar—that are ubiquitous in navigation systems precisely because they provide high-rate
dynamics information unavailable from position sensors alone. The goal of this project is to extend certifiable estimation to inertial navigation systems by devising suitable semidefinite relaxations for velocity and acceleration measurements.
The core technical challenge lies in higher-order nonconvex coupling: inertial measurements in the body frame must be transformed to the global frame via rotation matrices, then related to velocity/acceleration states. This introduces polynomial constraints of degree three or higher (rotations are quadratic; dynamics add another layer), which resist the semidefinite lifting techniques that work for quadratic position-based problems. Success requires identifying exploitable structure—temporal smoothness, differential relationships on SE(3), or algebraic symmetries—enabling tight convex relaxations with provable global optimality guarantees.

Many important tasks in robotics require methods to estimate a set of unknown states from noisy relative measurements between them. These inference problems are generally formulated as the Maximum Likelihood Estimate(MLE) problem. Recent techniques like SE-Sync have shown that it is possible to solve these problems to provable global optimality by formulating them as semidefinite programs. For safety-critical applications like autonomous aircraft, spacecraft navigation, or surgical robotics, this certification capability is extremely helpful; rather than blindly trusting an optimizer that may have failed, we obtain mathematical guarantees about solution quality.
However, current certifiable estimation frameworks, including SE-Sync cannot directly handle velocity and acceleration measurements from inertial sensors like gyroscopes, accelerometers, or Doppler radar. The core technical challenge lies in higher-order nonconvex coupling: inertial measurements in the body frame must be transformed to the global frame via rotation matrices, then related to velocity/acceleration states. The goal of this project is to extend certifiable estimation into the realm of inertial navigation systems, by devising suitable convex (semidefinite) relaxations of velocity and acceleration measurements. This requires both deep understanding of
the geometric structure of motion (Lie theory, manifold optimization) and sophisticated tools from convex algebraic geometry (moment relaxation hierarchies, sum-of-squares programming) to determine which relaxation strategies can provably recover global solutions despite the increased complexity.
'''