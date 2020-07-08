import sys
sys.path.append('/Users/umeco/projects/zero_DL2/src/')

import numpy as np

import pickle
from common.trainer import trainer
from common.optimizer import Adam
from cbow import CBOW
from commonl.util import create_contexts_target
from dataset import ptb