import os
from config import ROOT_DIR
from types import SimpleNamespace

os.environ["SM_CHANNEL_TRAINING"] = os.path.join(ROOT_DIR, ".rawdata", "acorn_sat", "sagemaker")
os.environ["SM_MODEL_DIR"] = os.path.join(ROOT_DIR, ".models")

from train import train

args = SimpleNamespace()
args.epochs = 10
args.batchsize = 128
args.lr = 0.01
train(args)