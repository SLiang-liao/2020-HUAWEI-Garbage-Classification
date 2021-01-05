class Config(object):
    backbone = 'densenet121_改分类器'
    num_classes = 43 #
    loss = 'CrossEntropyLoss'#focal_loss/CrossEntropyLoss
    #
    input_size = 224
    train_batch_size = 40 # batch size
    val_batch_size = 40
    test_batch_size = 1
    optimizer = 'sgd'
    lr = 1e-3 # adam 0.00001
    MOMENTUM = 0.9
    device = "cuda"  # cuda  or cpu
    gpu_id = [0,1]
    num_workers = 8  # how many workers for loading data
    max_epoch = 30
    lr_decay_epoch = 10
    lr_decay = 0.98  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    val_interval = 1
    print_interval = 100
    save_interval = 2
    min_save_epoch=2
    #
    log_dir = 'log/'
    train_val_data = './dataset/train_data'
    raw_json = 'garbage_classify_rule.json'
    train_list='./dataset/train.txt'
    val_list='./dataset/val.txt'
    #
    checkpoints_dir = './ckpt/'
