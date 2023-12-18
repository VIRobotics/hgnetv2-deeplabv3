class LabConfig(object):
    init_lr=7e-3
    min_lr_mutliply=0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_type = 'cos'


class UNetConfig(object):
    init_lr=1e-4
    min_lr_mutliply=0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'


class PSPNetConfig(object):
    init_lr = 1e-2
    min_lr_mutliply = 0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_type = 'cos'


class SegFormerConfig(object):
    init_lr = 1e-4
    min_lr_mutliply = 0.01
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2
    lr_decay_type = 'cos'


class HarDNetConfig(object):
    init_lr = 0.02
    min_lr_mutliply = 0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 5e-4
    lr_decay_type = 'cos'