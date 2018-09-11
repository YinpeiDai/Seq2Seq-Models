import argparse
import os
import warnings

# ignore all kinds of warnings for printing simplicity
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from train_manager import Train_BasicEncoderDecoder,Train_AttenNet, Train_CopyNet

# parse the cmd arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,
                    choices=[
                        'BasicSeq2Seq', 'AttenNet', 'CopyNet'
                    ])
args = parser.parse_args()
model = args.model


if model == 'BasicSeq2Seq':
    manager = Train_BasicEncoderDecoder()
    manager.run_model()
elif model == 'AttenNet':
    manager = Train_AttenNet()
    manager.run_model()
else:
    manager = Train_CopyNet()
    manager.run_model()