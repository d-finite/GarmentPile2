import argparse
import numpy as np

def parse_args_record():
    parser=argparse.ArgumentParser()
    
    def str_to_ndarray(value):
        value = value.strip('"').strip("\\")
        value_list=value.split(',')
        if len(value_list)!=3:
            raise argparse.ArgumentTypeError("The length of pos and ori must be 3")
        return np.array(value_list).astype(float)
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument("--ground_material_usd", type=str, default=None, help="ground material usd path")
    parser.add_argument("--data_collection_flag",type=str2bool, default=False, help="data collection flag")
    parser.add_argument("--record_vedio_flag",type=str2bool, default=False, help="record vedio flag")
    parser.add_argument("--env_random_flag", type=str2bool, default=False, help="env random flag")
    parser.add_argument("--garment_random_flag", type=str2bool, default=False, help="garemnt random flag")
    
    return parser.parse_args()

def parse_args_val():
    parser=argparse.ArgumentParser()
    
    def str_to_ndarray(value):
        value = value.strip('"').strip("\\")
        value_list=value.split(',')
        if len(value_list)!=3:
            raise argparse.ArgumentTypeError("The length of pos and ori must be 3")
        return np.array(value_list).astype(float)
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    def str2int(v):
        try:
            return int(v)
        except ValueError:
            raise argparse.ArgumentTypeError('Integer value expected.')
    
    parser.add_argument("--ground_material_usd", type=str, default=None, help="ground material usd path")
    parser.add_argument("--validation_flag",type=str2bool, default=False, help="validation flag")
    parser.add_argument("--record_vedio_flag",type=str2bool, default=False, help="record vedio flag")
    parser.add_argument("--env_random_flag", type=str2bool, default=False, help="env random flag")
    parser.add_argument("--garment_random_flag", type=str2bool, default=False, help="garemnt random flag")
    
    parser.add_argument("--training_data_num", type=str2int, default=50, help="training data number")
    parser.add_argument("--stage_1_checkpoint_num", type=str2int, default=1000, help="Stage 1 checkpoint number")
    parser.add_argument("--stage_2_checkpoint_num", type=str2int, default=1500, help="Stage 2 checkpoint number")
    parser.add_argument("--stage_3_checkpoint_num", type=str2int, default=1500, help="Stage 3 checkpoint number")
    
    return parser.parse_args()