# main_DDQL.py
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent_DDQL import Agent_DDQL
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from datetime import datetime
from utils.utils import *
from utils.config import *
import argparse
from PIL import Image


def train(transitions):
    parser = argparse.ArgumentParser(description="Super Mario RL Training")
    parser.add_argument('--job_id', type=int, required=True, help='Unique Job ID')
    args = parser.parse_args()
    job_id = args.job_id

    # Test
    # job_id = 5

    if torch.cuda.is_available():
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available")

    type = "DDQL"

    env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)

    timestamp = datetime.now().strftime('%m%d_%H%M%S')

    frames_dir, checkpoints_dir, log_filename, transitions_filename = (
        create_save_files_directories(timestamp, job_id, type))

    agent = Agent_DDQL(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

    # if not SHOULD_TRAIN:
    #     folder_name = ""
    #     ckpt_name = ""
    #     agent.load_model(os.path.join("models", folder_name, ckpt_name))
    #     agent.epsilon = 0.2
    #     agent.eps_min = 0.0
    #     agent.eps_decay = 0.0

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)

    for i in range(NUM_EPISODES):
        print("Episode:", i)
        state, _ = env.reset()

        episode_start_time = time.time()
        frame_counter = 0
        total_reward = 0
        done = False
        start_level = info.get('level', 'unknown')  # Replace 'level' with the correct key if different
        current_level = start_level

        while not done:
            frame_counter += 1
            a = agent.choose_action(state)
            new_state, reward, done, truncated, info  = env.step(a)
            print(new_state.shape)
            total_reward += reward

            current_level = info.get('level', current_level)  # Update current_level if 'level' is present

            if SHOULD_TRAIN:
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()

            state = new_state

            if frame_counter % SAVE_INTERVAL == 0:
                frame = env.render()  # No arguments needed
                if frame is not None:
                    try:
                        image = Image.fromarray(frame)
                        image_filename = os.path.join(frames_dir, f"episode_{i}_step_{frame_counter}_level_{current_level}.png")
                        image.save(image_filename)
                    except Exception as e:
                        print(f"Error saving frame: {e}")

        end_level = current_level

        # Calculate episode duration
        episode_duration = time.time() - episode_start_time

        # Log episode statistics
        episode_data = [i, total_reward, episode_duration, start_level, end_level]
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(episode_data)

        if (i + 1) % SAVE_INTERVAL == 0:
            save_model(agent.online_network, f"{checkpoints_dir}/mario_ddqn_{i}.pth")

        print(f"Episode {i}: Total Reward = {total_reward}, Duration = {episode_duration:.2f}s, Start Level = {start_level}, End Level = {end_level}", "Size of replay buffer = ", len(agent.replay_buffer), "Learn step counter = ", agent.learn_step_counter)

    env.close()

if __name__ == "__main__":
    train(transitions=False)