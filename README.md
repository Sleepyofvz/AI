# AI

pip install ftfy regex tqdm\
pip install git+https://github.com/openai/CLIP.git \
pip install spotipy

#moodmodel.pth CNN모듈 
import os
import glob
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

