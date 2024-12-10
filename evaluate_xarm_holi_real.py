from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait

import pyrealsense2 as rs
import torch
import time
import numpy as np
import threading
from tqdm import tqdm
import cv2
# Import xArm API
from xarm.wrapper import XArmAPI

class RealSenseStreamer:
    def __init__(self, num_cameras=3, show_window=False):
        # Configure depth and color streams for three cameras
        self.num_cameras = num_cameras
        self.pipelines = [rs.pipeline() for _ in range(num_cameras)]
        self.configs = [rs.config() for _ in range(num_cameras)]
        self.show_window = show_window

        # Enable all cameras by serial number
        ctx = rs.context()
        devices = ctx.query_devices()
        assert len(devices) >= self.num_cameras, 'Found {} cameras, but required {}'.format(len(devices), self.num_cameras)

        self.serial_numbers = [devices[i].get_info(rs.camera_info.serial_number) for i in range(num_cameras)]
        for i in range(num_cameras):
            self.configs[i].enable_device(self.serial_numbers[i])
            self.configs[i].enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def start_streams(self):
        # Start streaming from all three cameras
        for i in range(self.num_cameras):
            self.pipelines[i].start(self.configs[i])

    def stop_streams(self):
        # Stop streaming from all three cameras
        for pipeline in self.pipelines:
            pipeline.stop()

class XArmEvaluator:
    def __init__(self, arm_ip, ckpt_path, num_cameras=3, device='cuda', fps=30, inference_time_s=60):
        self.arm_ip = arm_ip
        self.ckpt_path = ckpt_path
        self.num_cameras = num_cameras
        self.device = device
        self.fps = fps
        self.inference_time_s = inference_time_s

        # Configuration and state from xarm_dualsense.py
        self.homing_position = [330, 0, 250, 180, 0, 0, 850]  # x, y, z, roll, pitch, yaw, gripper
        self.current_speed = 60
        self.gripper_position = self.homing_position[6]
        self.prev_gripper_position = self.gripper_position
        self.limits = {'x': 200, 'y': 200, 'z': 180, 'roll': 90, 'pitch': 90, 'yaw': 90}

        # Frame buffers and threading for camera streaming
        self.CAMERA_NAMES = [
            'observation.image.cam_left',
            'observation.image.cam_right',
            'observation.image.cam_wrist'
        ]
        self.frame_buffers = {name: None for name in self.CAMERA_NAMES}
        self.frame_lock = threading.Lock()
        self.is_streaming = False

        # Initialize devices
        self._initialize_arm()
        self._load_policy()
        self._initialize_cameras()

    def _initialize_arm(self):
        # Initialize the xArm
        self.arm = XArmAPI(self.arm_ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        # Move to homing position at the start
        self.move_to_home()
        print("Moved to HOME position")

    def _initialize_cameras(self):
        # Initialize RealSenseStreamer
        self.streamer = RealSenseStreamer(num_cameras=self.num_cameras, show_window=False)
        self.start_camera_stream()


    def _load_policy(self):
        # Load the pretrained policy from a local checkpoint
        self.policy = ACTPolicy.from_pretrained(self.ckpt_path)
        self.policy.to(self.device)

    def move_to_home(self):
        self.arm.set_position(
            x=self.homing_position[0],
            y=self.homing_position[1],
            z=self.homing_position[2],
            roll=self.homing_position[3],
            pitch=self.homing_position[4],
            yaw=self.homing_position[5],
            speed=self.current_speed,
            is_radian=False,
            wait=True
        )
        # Set gripper to homing position
        self.arm.set_gripper_position(self.gripper_position, wait=True)
        print("Gripper set to home position")

    def start_camera_stream(self):
        self.is_streaming = True
        self.streamer.start_streams()
        self.camera_thread = threading.Thread(target=self._camera_stream)
        self.camera_thread.start()

    def stop_camera_stream(self):
        self.is_streaming = False
        self.camera_thread.join()

    def _camera_stream(self):
        while self.is_streaming:
            frames = [pipeline.wait_for_frames() for pipeline in self.streamer.pipelines]
            color_frames = [frames[i].get_color_frame() for i in range(self.streamer.num_cameras)]
            if all(color_frames):
                color_images = [np.asanyarray(frame.get_data()) for frame in color_frames]
                with self.frame_lock:
                    for i, image in enumerate(color_images):
                        self.frame_buffers[self.CAMERA_NAMES[i]] = image

    def get_observation(self):
        # Capture observation from frame buffers
        observation = {}
        with self.frame_lock:
            for i in range(self.num_cameras):
                frame = self.frame_buffers.get(self.CAMERA_NAMES[i])
                if frame is not None:
                    # Convert image to torch tensor
                    image_tensor = torch.from_numpy(frame).type(torch.float32) / 255
                    image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # Channel first
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
                    observation[self.CAMERA_NAMES[i]] = image_tensor
                else:
                    # If frame is not available, use a zero tensor or handle appropriately
                    observation[self.CAMERA_NAMES[i]] = torch.zeros((1, 3, 480, 640), device=self.device)

        # Read the arm's current state
        code, arm_state = self.arm.get_position(is_radian=False)
        # Include gripper state
        arm_state_with_gripper = arm_state + [self.gripper_position]
        arm_state_tensor = torch.tensor(arm_state_with_gripper, dtype=torch.float32).unsqueeze(0).to(self.device)
        observation['observation.state'] = arm_state_tensor

        return observation

    def run_inference(self):
        total_steps = int(self.inference_time_s * self.fps)
        
        axes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        min_limits = [self.homing_position[idx] - self.limits[axis] for idx, axis in enumerate(axes)]
        max_limits = [self.homing_position[idx] + self.limits[axis] for idx, axis in enumerate(axes)]

        time.sleep(1)
        for step in tqdm(range(total_steps)):
            start_time = time.perf_counter()
            
            observation = self.get_observation()
            
            action = self.policy.select_action(observation)
            
            action = action.squeeze(0).to("cpu").numpy()
            action = np.array(action, dtype=np.float32)
            
            gripper_position = action[6]
            gripper_position = np.clip(gripper_position, 0, 850)
            gripper_position = np.round(gripper_position / 85) * 85
            
            for idx in range(6):
                action[idx] = np.clip(action[idx], min_limits[idx], max_limits[idx])

            # # # Save the observation's images
            # for i in range(self.num_cameras):
            #     image_tensor = observation[self.CAMERA_NAMES[i]].squeeze(0)
            #     image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255  # Convert to HxWxC and scale to [0, 255]
            #     image_np = image_np.astype(np.uint8)
            #     cv2.imwrite(f'tmp_evaluate_observation_{i}.png', image_np)

            # if step%5 == 0:
            #     breakpoint()
            self.arm.set_position(
                x=action[0],
                y=action[1],
                z=action[2],
                roll=180, 
                pitch=action[4],
                yaw=action[5],
                speed=self.current_speed,
                is_radian=False,
                radius=0,
                wait=False,
                relative=False 
            )
            
            if gripper_position != self.prev_gripper_position:
                self.arm.set_gripper_position(gripper_position, wait=False)
                self.prev_gripper_position = gripper_position
            
            dt_s = time.perf_counter() - start_time
            busy_wait(max(0, 1 / self.fps - dt_s))

    def cleanup(self):
        # Stop streaming
        self.stop_camera_stream()
        self.streamer.stop_streams()
        self.arm.disconnect()

    def run(self):
        try:
            self.run_inference()
        finally:
            self.cleanup()

if __name__ == "__main__":
    evaluator = XArmEvaluator(
        arm_ip='192.168.0.211',
        ckpt_path="jhseon-holiday/act_xarm_holi_test",
        num_cameras=3,
        device="cuda",
        fps=30,
        inference_time_s=60
    )
    evaluator.run()