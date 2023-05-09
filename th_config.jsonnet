// DataSet
local class1_Bronchial = {
    classes: ["Bronchial"],
    train_uid_file: "/home/oem/dlc/Data/500_100/train.txt",
    valid_uid_file: "/home/oem/dlc/Data/500_100/valid.txt",
    base_path: "/home/oem/dlc/Data/",
};

local class1_Sphere = {
    classes: ["Sphere"],
    train_uid_file: "/home/oem/dlc/Data/500_100/train.txt",
    valid_uid_file: "/home/oem/dlc/Data/500_100/valid.txt",
    base_path: "/home/oem/dlc/Data/",
};


// 使用这个返回{}的函数检查args是否正常
local checkArgs(args) =
  // TODO: check loss_function
  assert std.member([
    "dice",
    "bce",
    "focal",
    "wbce",
    "dice-focal",
    "dice-bce",
  ], args.loss_function): "=======" + "loss: " + args.loss_function + " is invalid======";
  assert std.member([
    "rms",
    "sgd",
    "adam",
    "adamW",
  ], args.optimizer_name): "=======" + "opt: " + args.optimizer_name + " is invalid======";
  assert std.member([
    "stepLR",
    "exponentialLR",
    "customLR1",
    "OneCycleLR",
  ], args.scheduler_name): "=======" + "scheduler: " + args.scheduler_name + " is invalid======";
  // return
  {};


local switchDataset(classname, info={}) = if std.member(["Sphere", "Bronchial"], classname) then {
  "Bronchial": class1_Bronchial,
  "Sphere": class1_Sphere,
  // 下面一步直接做到null就不要了
}[std.toString(classname)] + std.prune(info)
else
  // 如果classNumber是已经定义好的数目，那么可以用这些作为默认值，否则还是自己输入吧
  info;


local stragety(
    transname="clip-rotated", 
    loss_function="bce", 
    optimizer_name="adam", 
    scheduler_name="customLR1",
    output_stride="32",
) = transname + "-" + loss_function + "-" + optimizer_name + "-" + scheduler_name + "-" + output_stride + "x";


local save_path(
    base_path="./output", 
    name="pth", 
    classname="Sphere",
    encoder_name="stdc2", 
    decoder_name="Unet", 
    stragety="clip-rotated",
) = base_path  + "/" + name + "/" +classname + "/" + encoder_name + "_" + decoder_name + "/" + stragety;


function(
    // device
    device = "cuda:0",

    //ww,wl
    windowlevel=-850,
    windowwidth=310,

    // downsampling
    output_stride=32,

    //input channel
    in_channels=1,

    // encoder and decoder
    encoder_name="stdc2",
    decoder_name="Unet",

    // train/valid data 
    classname="Bronchial",
    classes=null,
    train_uid_file=null,
    valid_uid_file=null,
    base_path=null,

    // dataloader params
    shuffle=true,  // 是否需要打乱数据
    num_workers=8,  // 多线程加载所需要的线程数目
    pin_memory=true,  // 数据从CPU->pin_memory—>GPU加速
    batch_size=16,  // 20
    val_batch_size=8,

    // patch_size
    middle_patch_size=512,  // 512 # 224 #   #

    // 这两个参数是env
    // env = time.strftime(tfmt) + "resnet50_lr0.01_bs20_adam_sunwa"  # Visdom env

    plot_every=50,
    weight_decay=5e-4,  // 5e-4

    // RMS
    momentum=0.9,
    eps=1,  // DeepPATH  RMSPROP_EPSILON = 1.0

    gamma=0.16,  // learning_rate_decay_factor", 0.16  StepLR
    step_size=30,  //没经过30个epoch，lr衰减一次

    max_epoch=80,
    lr=1e-4,  // 学习率
    min_lr=1e-10,  // 当学习率低于这个值，就退出训练
    lr_decay=0.5,  // 当一个epoch的损失开始上升lr = lr*lr_decay


    loss_function="focal",  // 损失函数,对应于models.loss.py中的函数名
    pretrained_model=null,

    // new configs
    is_training=null,  // 影响input_config2.Config的逻辑

    // optimizer_name
    optimizer_name="adam",

    // scheduler_name
    scheduler_name="customLR1",

    // git commit for debug purpose
    git_commit="not-specified",
) {
    classInput: switchDataset(classname, {
        classes: classes,
        train_uid_file: train_uid_file,
        valid_uid_file: valid_uid_file,
        base_path: base_path,
    },),
    args: {
        shuffle: shuffle,
        num_workers: num_workers,
        pin_memory: pin_memory,
        batch_size: batch_size,
        val_batch_size: val_batch_size,
        middle_patch_size: middle_patch_size,
        plot_every: plot_every,
        weight_decay: weight_decay,
        momentum: momentum,
        eps: eps,
        gamma: gamma,
        step_size: step_size,
        max_epoch: max_epoch,
        lr: lr,
        min_lr: min_lr,
        lr_decay: lr_decay,
        encoder_name: encoder_name,
        decoder_name: decoder_name,
        loss_function: loss_function,
        pretrained_model: pretrained_model,
        device: device,

        // 对于optimizer的选择
        optimizer_name: optimizer_name,
        // lr衰减策略
        scheduler_name: scheduler_name,
        windowlevel: windowlevel,
        windowwidth: windowwidth,
        output_stride: output_stride,
        stragety: stragety(
                            optimizer_name=optimizer_name, 
                            loss_function=loss_function, 
                            output_stride=output_stride
                    ),

        in_channels: in_channels,
        pth_save_base_path: save_path(
                            stragety=self.stragety, 
                            name="pth",
                            encoder_name=self.encoder_name, 
                            decoder_name=self.decoder_name,
                        ),
        log_save_base_path: save_path(
                            stragety=self.stragety, 
                            name="logs", 
                            encoder_name=self.encoder_name, 
                            decoder_name=self.decoder_name,
                        ),
    },
    errors: checkArgs(self.args),

    otherInfo: {
        code: "th",
    },

    debugInfo: {
    "   --debug--": {
        git_commit: git_commit,
        classname: classname,
    },
    },
    final: self.args + self.classInput + self.errors + self.debugInfo,
    // + self.otherInfo 
}.final
