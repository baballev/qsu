import os
from datetime import datetime
import pickle
import torch
import bz2
import random

from models import TaikoNetwork
from memory import ReplayMemory2

MODEL_SUFFIX = '2fclayers'
OPTIMIZER_SUFFIX = 'optim'
MEMORY_SUFFIX = 'memory'

class Checkpointer:
    def __init__(self, root_dir='./weights/Taiko/', beatmap_name=None):
        date = str(datetime.date(datetime.now()))
        if beatmap_name is None:
            self.bn = ''
        else:
            self.bn = beatmap_name

        self.memory_name = MEMORY_SUFFIX + '.pckl'
        self.weights_name = MODEL_SUFFIX + '.pt'
        self.opti_name = OPTIMIZER_SUFFIX + '.pt'

        # Detect if folder is for loading previous training or a new one
        if os.path.exists(root_dir):
            if len(os.listdir(root_dir)) == 0:
                self.loading = False
            else:
                if len(os.listdir(root_dir)) == 3:
                    if os.path.exists(os.path.join(root_dir, self.memory_name)):
                        self.loading = True
                    else:
                        self.loading = False
                else:
                    self.loading = False
        else:
            os.mkdir(root_dir)
            self.loading = False

        if not self.loading:
            self.curr_root = os.path.join(root_dir, date + '_' + self.bn )
            n = len(os.listdir(root_dir))
            while os.path.exists(self.curr_root + str(n)):
                n += 1
            self.curr_root += str(n)
            os.mkdir(self.curr_root)
            self.count = 0
        else:
            self.curr_root = os.path.join(root_dir, '../')
            self.count = 0
            self.load_dir = root_dir
        return

    def save(self, memory, model, optimizer):
        while os.path.exists(os.path.join(self.curr_root, str(self.count) + '/')):
            self.count += 1
        tmp = os.path.join(self.curr_root, str(self.count) + '/')
        os.mkdir(tmp)
        with bz2.open(tmp + self.memory_name, 'wb') as f:
            pickle.dump(memory, f, protocol=4)
        torch.save(model.state_dict(), tmp + self.weights_name)
        torch.save(optimizer.state_dict(), tmp + self.opti_path)
        self.count += 1
        print("Saved model + memory + optimizer to: " + tmp)
        return

    def load(self):
        if self.loading:
            load_memory = os.path.join(self.load_dir, self.memory_name)
            load_optim = os.path.join(self.load_dir, self.opti_name)
            load_network = os.path.join(self.load_dir, self.weights_name)
            return load_memory, load_optim, load_network
        else:
            return None, None, None

