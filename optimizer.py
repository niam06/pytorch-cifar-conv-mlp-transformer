import torch.optim as optim

def Optimizer(net, opt, args):
    if opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    elif opt == "rmsprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    elif opt == "adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr)
    elif opt == "adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    elif opt == "adamax":
        optimizer = optim.Adamax(net.parameters(), lr=args.lr)
    elif opt == "asgd":
        optimizer = optim.ASGD(net.parameters(), lr=args.lr)
    elif opt == "lbfgs":
        optimizer = optim.LBFGS(net.parameters(), lr=args.lr)
    else:
        raise ValueError("Optimizer not supported")
    
    return optimizer