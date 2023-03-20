import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import numpy as np

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets import(
    OxfordPetDataset, 
    SimpleOxfordPetDataset,
    SegDataset,
    SegDataset1
)
from segmentation_models_pytorch.utils.train import(
    Epoch, TrainEpoch, ValidEpoch
)
# from segmentation_models_pytorch.losses import (
#     DiceLoss,
#     JaccardLoss,
#     SoftBCEWithLogitsLoss,
#     SoftCrossEntropyLoss,
#     TverskyLoss,
#     MCCLoss,
# )
from segmentation_models_pytorch.utils.losses import(
    JaccardLoss,
    DiceLoss,
    BCELoss
)
from segmentation_models_pytorch.utils.metrics import(
    IoU, Fscore, Precision, Recall, Accuracy
)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    # plt.show()
    # time.sleep(1)
    # plt.close()
    plt.pause(1)


def test(test_dataset, **entrance):
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        entrance["batch_size"], ##batch_szie
        num_workers=entrance["num_workers"],
        shuffle=False,
        pin_memory=entrance["pin_memory"],
        drop_last=False,
    )

    encoder_name = entrance["encoder_name"]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
    best_model = smp.Unet(
        encoder_name=encoder_name,  
        encoder_weights=None, 
        in_channels=1, 
        classes=1,
    ).to(device)

    # evaluate model on test set
    criterion = BCELoss()
    metrics = [IoU(), Fscore(), Precision(), Recall(), Accuracy()]
    state_dict = torch.load(entrance["best_model"])
    best_model.load_state_dict(state_dict)

    # test_epoch = smp.utils.train.ValidEpoch(
    #     model=best_model,
    #     loss=criterion,
    #     metrics=metrics,
    #     device=device,
    # )
    # logs = test_epoch.run(test_dataloader)

    # import pdb; pdb.set_trace()

    # best_model.eval()
    # for i in range(len(test_dataset)): ##从28个slice开始有肺
    # for i in range(100, 120): #28, 40
    for i in range(0, 291, 30):
    # for i in range(30,34):
        n = i
        # n = np.random.choice(len(test_dataset))
        
        # image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0).float()
        pr_mask = best_model.predict(x_tensor)

        pr_mask = torch.nn.functional.sigmoid(pr_mask)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        image = image.squeeze().astype(np.uint8) 
        params = {"image": image, "ground_truth_mask": gt_mask, "predicted_mask": pr_mask}
        visualize(**params)


if __name__ == "__main__":
    entrance = {
        "encoder_name": "efficientnet-b4",
        "best_model": ".\\output\\pth\\efficientnet-b4_noclip\\efficientnet-b4_epoch_23.pth",
        # "best_model": ".\\output\\pth\\efficientnet-b4\\0316_174217\\efficientnet-b4_epoch_14.pth",
        # "best_model": ".\\output\\pth\\efficientnet-b4\\0315_clip\\efficientnet-b4_epoch_23.pth",
        # "encoder_name": "mobileone_s4",
        # "best_model": ".\\output\\pth\\mobileone_s4\\mobileone_s4_epoch_8.pth",
        "device": "cuda:0",
        "test_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val",
        "windowlevel": -600,
        "windowwidth": 2000,
        "middle_patch_size": 512,
        "in_channels": 1,
        "num_workers": 8,  # 多线程加载所需要的线程数目
        "pin_memory": True,  # 数据从CPU->pin_memory—>GPU加速
        "batch_size": 4,
    } 


    test_dataset = smp.datasets.SegDataset1(
        base_path=entrance["test_base_path"],
        height=entrance["middle_patch_size"],
        width=entrance["middle_patch_size"],
        channels=entrance["in_channels"],
        windowlevel=entrance["windowlevel"],
        windowwidth=entrance["windowwidth"],
        transform=None
    )
    test(test_dataset, **entrance)
    print("Happy End!")