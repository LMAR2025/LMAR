import json
import nltk
from pathlib import Path
import numpy as np
import pandas as pd
import os
import sys
import re
from tqdm import tqdm
import random
import copy
import io
from pprint import pprint
import asyncio
import ast
import time
import concurrent
import argparse
import pickle
import gc
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import random_split

from openai import OpenAI
import openai

from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig

from langchain_core.documents import Document


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")