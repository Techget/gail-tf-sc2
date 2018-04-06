from gailtf.baselines import logger
import pickle as pkl
import numpy as np
from tqdm import tqdm
import ipdb
import os
from google.protobuf.json_format import MessageToJson

class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()
       
    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels

class SC2Dataset(object):
    def __init__(self, expert_path, train_fraction=0.7, ret_threshold=None, traj_limitation=np.inf, randomize=True):
        self.map_used = 'Odyssey LE'
        self.race_used = 'Terran'

        self.replay_files = []
        for file in os.listdir(expert_path):
            if file.endswith(".p"):
                self.replay_files.append(file)

        self.replay_files_index = 0
        self.loaded_replay = None
        self.loaded_replay_pointer = 0
        self.win_player_id = None

    def get_next_batch(self, batch_size, split=None):
        while loaded_replay == None:
            self.loaded_replay = pickle.load(open(self.replay_files[self.replay_files_index], "rb"))
            loaded_replay_info_json = MessageToJson(self.loaded_replay['info'])

            if loaded_replay_info_json['mapName'] != self.map_used or \
                loaded_replay_info_json['player_info'][0]['playerInfo']['raceActual'] != self.race_used or \
                loaded_replay_info_json['player_info'][1]['playerInfo']['raceActual'] != self.race_used:
                self.loaded_replay = None
                self.replay_files_index += 1
                continue

            self.replay_files_index += 1
            self.loaded_replay_pointer = 0
            self.win_player_id = int(loaded_replay_info_json['player_info'][1]['playerResult']['playerId']) if\
                loaded_replay_info_json['player_info'][0]['playerResult']['result'] == 'Victory' else \
                int(loaded_replay_info_json['player_info'][0]['playerResult']['playerId'])

        obs = []
        acs = []
        for i in range(self.loaded_replay_pointer, len(self.loaded_replay['state'])):
            if len(obs) >= batch_size:
                break
            self.loaded_replay_pointer += 1
            temp_obs = []
            temp_acs = []

            if self.loaded_replay['state'][i]['player'][0] == self.win_player_id:
                if len(self.loaded_replay['state'][i]['actions']) == 0:
                    continue

                for x in self.loaded_replay['state'][i]['minimap']
                    temp_obs.extend(list(x.flatten()))

                for x in self.loaded_replay['state'][i]['screen']:
                    temp_obs.extend(list(x.flatten()))

                temp_obs.extend(list(self.loaded_replay['state'][i]['player']))
                temp_obs.extend(list(self.loaded_replay['stete'][i]['available_actions']))

                for a in self.loaded_replay['state'][i]['actions']:
                    # one captured state, may have multiple actions, so output should be the
                    # same observation with different action ids
                    obs.append(temp_obs)
                    acs.append(a[0])

        if(self.loaded_replay_pointer == len(self.loaded_replay['state'])) - 1:
            self.loaded_replay = None
            self.loaded_replay_pointer = 0
            self.win_player_id = None

        return obs, acs











