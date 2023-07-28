from distutils import core
from email.policy import default

from numpy import histogram
import coreset_selection
from data_reweighting.meta_weight_net import meta_weight_net, get_args
from data import get_classes_count, DATASETS_DICT, get_transforms, get_dataset, CustomSubset
from utils import set_cudnn, set_seed, get_model, broadcast_weights, get_histogram, log_on_wandb
from args import parse_args
import torch
from train import train
import wandb

def main():
    args = parse_args()
    
    print(args)
    
    tags = []
    
    if args.weighting_method == 'uniform':
        args.coreset_method = '-'
        args.coreset_ratio = 1
        args.feature_extractor = '-'
        tags = [args.arch, args.dataset, args.weighting_method]
    else:
        tags = [args.arch, args.dataset, args.weighting_method, args.coreset_method]
        
    try:
        wandb.login(key=args.wandb_api)
        wandb.init(
            # set the wandb project where this run will be logged
            project="OPTML",
            config={
                "arch": args.arch,
                "feature_extractor": args.feature_extractor,
                "weighting_method": args.weighting_method,
                "coreset_method": args.coreset_method,
                "coreset_ratio": args.coreset_ratio,
                "batch_size": args.batch_size,
                "optmizer": 'SGD',
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.wd,
                "milestone_ratios": args.milestone_ratios,
                "test_interval": args.test_interval,
                "device": args.device,
                "seed": args.seed,
                "dataset": args.dataset,
                "epochs": args.epochs,
            },
            tags=tags
        )
    except:
        print("Failed to Login to WANDB!")
    
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
    
    _, imagenet_transform = get_transforms('imagenet')
    train_dataset_imagenet = get_dataset(root=args.data_dir, dataset_name=args.dataset, train=True, transform=imagenet_transform)
    
    train_transform, test_transform = get_transforms(args.dataset)
    trainset = get_dataset(root=args.data_dir, dataset_name=args.dataset, train=True, transform=train_transform)
    testset = get_dataset(root=args.data_dir, dataset_name=args.dataset, train=False, transform=test_transform)
    
    count = int(args.coreset_ratio * len(train_dataset_imagenet))
    
    coreset_indices = torch.arange(len(trainset))
    
    if args.coreset_method == 'random':
        coreset_indices = coreset_selection.random_selection.RandomCoresetSelection(args.feature_extractor, train_dataset_imagenet, args.dataset, count, device=args.device).get_coreset()
    elif args.coreset_method == 'moderate_selection':
        coreset_indices = coreset_selection.moderate_selection.ModerateCoresetSelection(args.feature_extractor, train_dataset_imagenet, args.dataset, count, device=args.device).get_coreset()
    
    
    coreset_weights = None
    
    weighted_coreset_accuracy = '-'
    
    if args.weighting_method == 'meta_weight_net':
        default_args = get_args()
        default_args.dataset = args.dataset
        default_args.num_classes = get_classes_count(args.dataset)
        coreset = CustomSubset(trainset, coreset_indices)
        num_meta = max(int(0.2 * len(coreset)), 100)
        default_args.num_meta = num_meta
        coreset_weights, weighted_coreset_accuracy = meta_weight_net(default_args, coreset, testset)
    elif args.weighting_method == 'uniform':
        coreset_weights = torch.ones(coreset_indices.shape[0])
    else:
        raise NotImplementedError
        
    
    

    pre_info = {
        'weighted_coreset_accuracy': weighted_coreset_accuracy,
        'coreset_weights': get_histogram(coreset_weights)
    }
    
    model = get_model(args.arch, get_classes_count(args.dataset)).to(args.device)
    
    try:
        wandb.watch(model)
    except:
        print("Failed to watch model!")
        
    if args.weighting_method == 'uniform':
        pre_info['full_weights'] = get_histogram(torch.ones(len(trainset)))
        log_on_wandb(pre_info)
        train(model, trainset, testset, torch.ones(len(trainset)), args.batch_size, args.device, args.epochs, args.milestone_ratios, args.test_interval, args.lr, args.momentum, args.wd)
    elif args.weighting_method == 'meta_weight_net':
        coreset_imagenet = CustomSubset(train_dataset_imagenet, coreset_indices)
        weights = broadcast_weights(args.feature_extractor, coreset_weights, train_dataset_imagenet, coreset_imagenet, num_workers=int(args.num_workers), device=args.device, classwise=args.weight_assignment == 'classwise')
        pre_info['full_weights'] = get_histogram(weights)
        log_on_wandb(pre_info)
        train(model, trainset, testset, weights, args.batch_size, args.device, args.epochs, args.milestone_ratios, args.test_interval, args.lr, args.momentum, args.wd)
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main()