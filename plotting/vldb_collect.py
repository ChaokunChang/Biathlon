import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import os
import numpy as np
import math
import json
from matplotlib.transforms import Bbox
from typing import List
from tap import Tap

from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS

class VLDBArgs(Tap):
    pass