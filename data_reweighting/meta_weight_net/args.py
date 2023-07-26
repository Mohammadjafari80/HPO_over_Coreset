import argparse


def get_args():
    args = argparse.Namespace(
            device='cuda',
            meta_net_hidden_size=100,
            meta_net_num_layers=1,
            lr=0.1,
            momentum=0.9,
            dampening=0.0,
            nesterov=False,
            weight_decay=5e-4,
            meta_lr=1e-5,
            meta_weight_decay=0.0,
            dataset='cifar10',
            num_meta=1000,
            batch_size=100,
            max_epoch=120,
            meta_interval=1,
            paint_interval=20
        )
    
    return args