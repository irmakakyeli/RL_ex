import sys
import requests
import torch
import subprocess
import shutil
import os
import datetime
import tensorflow as tf

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gym.wrappers import FlattenObservation

repo_dir = "boptestGymService"
if not os.path.exists(repo_dir):
    # Clone the repository
    try:
        subprocess.run([
            "git", "clone",
            "-b", "boptest-gym-service",
            "https://github.com/ibpsa/project1-boptest-gym.git",
            repo_dir
        ], check=True)
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print("Error cloning repository:", e)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(BASE_DIR, "boptestGymService"))

from boptestGymService.boptestGymEnv import BoptestGymEnv
from boptestGymEnv import DiscretizedActionWrapper
from boptestGymEnv import DiscretizedObservationWrapper



# Redefine reward function
class BoptestGymEnvCustomReward(BoptestGymEnv):
    def get_reward(self):
        '''Custom reward function. We use the BOPTEST `GET /kpis` API call to compute the
        total cummulative discomfort from the beginning of the episode. Note
        that this is the true value that BOPTEST uses when evaluating
        controllers.

        'tdis_tot': temp discomfort
         'idis_tot': 0,
         'ener_tot': total energy
         'cost_tot': total cost
         'emis_tot': total emission
         'pele_tot': defines the HVAC peak electrical demand.
         'pgas_tot': defines the HVAC peak gas demand.
         'pdih_tot': defines the HVAC peak district heating demand.
         'time_rat': defines the average ratio between the controller computation time and the test simulation control step. The controller computation time is measured as the time between two emulator advances
        '''
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        objective_integrand = kpis['tdis_tot']
        # Give reward if there is not immediate increment in discomfort
        if objective_integrand == self.objective_integrand:
          reward=1
        else:
          reward=0
        # Record current objective integrand for next evaluation
        self.objective_integrand = objective_integrand
        return reward

def create_env(url, test_case):
    print("Creating the environment")
    lower_setp = 21 + 273.15
    upper_setp = 24 + 273.15

    env = BoptestGymEnvCustomReward(url=url,
                                    testcase=test_case,
                                    actions=['hvac_oveZonSupCor_TZonCooSet_u',
                                             'hvac_oveZonSupNor_TZonCooSet_u',
                                             'hvac_oveZonSupSou_TZonCooSet_u'],
                                    observations={"hvac_reaZonCor_TZon_y": (lower_setp, upper_setp),
                                                  "hvac_reaZonNor_TZon_y": (lower_setp, upper_setp),
                                                  "hvac_reaZonSou_TZon_y": (lower_setp, upper_setp)},
                                    random_start_time=False,
                                    start_time=154 * 24 * 3600,
                                    max_episode_length=24 * 3600,
                                    warmup_period=24 * 3600,
                                    step_period=900)
    print("Environment created")
    return env


def train_model(env, log_path, model_path):
    print("inside train model")
    model = TD3('MlpPolicy', env, verbose=1, learning_rate=0.001, gamma=0.99, batch_size=100,
                policy_delay=2)
    print("created the model")
    model.learn(total_timesteps=10)
    print("learning complete")
    model.save(model_path)
    evaluate_policy(model, env, n_eval_episodes=3)
    env.stop()

if __name__ == "__main__":
    print("inside main /////////////77")
    url = 'http://localhost:80'
    test_case = "multizone_office_simple_air"
    log_path = "local_files/Logs"
    model_path = "local_files/saved_models"

    env = create_env(url, test_case)
    train_model(env, log_path, model_path)

