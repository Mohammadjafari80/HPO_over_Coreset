import coreset_selection
from data_reweighting.meta_weight_net import meta_weight_net, get_args
from data import get_classes_count, DATASETS_DICT, get_transforms, get_dataset
from utils import set_cudnn, set_seed
from args import parse_args

def main():
    args = parse_args()
    print(args)
    
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
    
    
if __name__ == "__main__":
    main()