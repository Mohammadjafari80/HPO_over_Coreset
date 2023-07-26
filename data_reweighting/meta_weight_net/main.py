import torch.optim
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from data_reweighting.meta_weight_net.meta import *
from data_reweighting.meta_weight_net.model import *
from data_reweighting.meta_weight_net.datasets import *
from data_reweighting.meta_weight_net.utils import *


def meta_weight_net(args, trainset, testset):

    meta_net = MLP(hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers).to(device=args.device)
    net = ResNet32(args.num_classes).to(device=args.device)

    criterion = nn.CrossEntropyLoss().to(device=args.device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    lr = args.lr

    train_dataloader, meta_dataloader, test_dataloader, train_dataloader_unshuffled = build_dataloader(
        seed=args.seed,
        trainset=trainset,
        testset=testset,
        num_meta_total=args.num_meta,
        batch_size=args.batch_size,
    )

    meta_dataloader_iter = iter(meta_dataloader)

    for epoch in tqdm(list(range(args.max_epoch))):

        if epoch >= 80 and epoch % 20 == 0:
            lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        print('Training...')
        for iteration, (inputs, labels) in enumerate(train_dataloader):
            net.train()
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            if (iteration + 1) % args.meta_interval == 0:
                pseudo_net = ResNet32(args.num_classes).to(args.device)
                pseudo_net.load_state_dict(net.state_dict())
                pseudo_net.train()

                pseudo_outputs = pseudo_net(inputs)
                pseudo_loss_vector = functional.cross_entropy(pseudo_outputs, labels.long(), reduction='none')
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
                pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
                pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)

                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                del pseudo_grads

                try:
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    meta_inputs, meta_labels = next(meta_dataloader_iter)

                meta_inputs, meta_labels = meta_inputs.to(args.device), meta_labels.to(args.device)
                meta_outputs = pseudo_net(meta_inputs)
                meta_loss = criterion(meta_outputs, meta_labels.long())

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            outputs = net(inputs)
            loss_vector = functional.cross_entropy(outputs, labels.long(), reduction='none')
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

            with torch.no_grad():
                weight = meta_net(loss_vector_reshape)

            loss = torch.mean(weight * loss_vector_reshape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Computing Test Result...')
        test_loss, test_accuracy = compute_loss_accuracy(
            net=net,
            data_loader=test_dataloader,
            criterion=criterion,
            device=args.device,
        )


        print('Epoch: {}, (Loss, Accuracy) Test: ({:.4f}, {:.2%}) LR: {}'.format(
            epoch,
            test_loss,
            test_accuracy,
            lr,
        ))
        
        weight_array = []
        
        for iteration, (inputs, labels) in enumerate(train_dataloader_unshuffled):
            net.train()
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = net(inputs)
            loss_vector = functional.cross_entropy(outputs, labels.long(), reduction='none')
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
            
            with torch.no_grad():
                weight = meta_net(loss_vector_reshape)
                
            weight_array.extend(weight.cpu().numpy().squeeze().tolist())
            
        return weight_array