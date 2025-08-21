# utils.py

import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_output(inputs, outputs, targets, masks, save_path=None):
    inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
    outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)
    targets = targets.cpu().numpy().transpose(0, 2, 3, 1)
    masks = masks.cpu().numpy().transpose(0, 2, 3, 1)

    def denormalize(image):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        return image

    inputs = [denormalize(input[..., :3]) for input in inputs]
    outputs = [denormalize(output) for output in outputs]
    targets = [denormalize(target) for target in targets]

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        axs[i, 0].imshow(inputs[i])
        axs[i, 0].set_title('Input Image')
        axs[i, 0].axis('off')
        axs[i, 1].imshow(targets[i])
        axs[i, 1].set_title('Original Image')
        axs[i, 1].axis('off')
        axs[i, 2].imshow(masks[i].squeeze(), cmap='gray')
        axs[i, 2].set_title('Mask Image')
        axs[i, 2].axis('off')
        axs[i, 3].imshow(outputs[i])
        axs[i, 3].set_title('Output Image')
        axs[i, 3].axis('off')

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path)
    plt.close(fig)
