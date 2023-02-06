class MyGlobals(object):

    DATADIR = "data/"
    RESULTDIR = "./results"

    nl = -1
    num_seeds = 1
    sample_seed = 42
    cuda_id = 0
    hidden_size = 16
    num_hidden = 2
    attn_heads = 8

    # Training args
    lr = 0.05
    num_epochs = 500
    save_epoch = 100
    dropout = 0.1
    weight_decay = 5e-4
    optimizer = "sgd"
    momentum = 0.9

