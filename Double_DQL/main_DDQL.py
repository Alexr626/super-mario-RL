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


def train(transitions, job_id):
    type = "DDQL"
    torch.manual_seed(123)
    timestamp = datetime.now().strftime('%m%d_%H%M%S')

    if torch.cuda.is_available():
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available")

    env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb_array', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)

    frames_dir, checkpoints_dir, log_filename, transitions_filename = (
        create_save_files_directories(timestamp, job_id, type))

    agent = Agent_DDQL(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

    if not SHOULD_TRAIN:
        folder_name = ""
        ckpt_name = ""
        agent.load_model(os.path.join("models", folder_name, ckpt_name))
        agent.epsilon = 0.2
        agent.eps_min = 0.0
        agent.eps_decay = 0.0

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)

    start_time = time.time()

    for episode in range(NUM_EPISODES):
        print("Episode:", episode)
        state, _ = env.reset()

        episode_start_time = time.time()
        frame_counter = 0
        total_reward = 0
        done = False

        while not done:
            frame_counter += 1
            a = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(a)
            # print(new_state.shape)
            total_reward += reward

            if frame_counter == 1:
                start_world = str(info['world']) + "_" + str(info['stage'])  # Replace 'stage' with the correct key if different

            current_world = str(info['world']) + "_" + str(info['stage'])

            if SHOULD_TRAIN:
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()

            state = new_state

            if frame_counter % SAVE_INTERVAL == 0:
                frame = env.render()  # No arguments needed
                if frame is not None:
                    try:
                        image = Image.fromarray(frame)
                        image_filename = os.path.join(frames_dir, f"episode_{episode}_step_{frame_counter}_level_{current_world}.png")
                        image.save(image_filename)
                    except Exception as e:
                        print(f"Error saving frame: {e}")

        end_world = current_world

        # Calculate durations
        episode_duration = time.time() - episode_start_time
        total_duration = time.time() - start_time

        # Log episode statistics
        episode_data = [episode, total_reward, episode_duration, total_duration, start_world, end_world]
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(episode_data)

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_model(agent.policy_net, os.path.join(checkpoints_dir, f"/mario_dqn_{episode}.pth"))

        print(f"Episode {episode}: Total Reward = {total_reward}, Episode Duration = {episode_duration:.2f}s, "
              f"Total Duration = {total_duration:.2f}s, Start stage = {start_world}, End stage = {end_world}",
              "Size of replay buffer = ", len(agent.replay_buffer), "Learn step counter = ", agent.learn_step_counter)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Mario RL Training")
    parser.add_argument('--job_id', type=int, required=True, help='Unique Job ID')
    args = parser.parse_args()
    job_id = args.job_id

    # Test
    # job_id = 5

    train(transitions=False, job_id=job_id)