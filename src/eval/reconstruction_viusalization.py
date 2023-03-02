from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import torch


def _mse(imageA: np.ndarray, imageB: np.ndarray):

    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(
    images: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    predictions: list[torch.Tensor],
    actuals: list[torch.Tensor],
):

    # s = ssim(imgAnp, imgBnp)
    # setup the figure
    fig = plt.figure("Comparison", figsize=(3, len(images)))
    fig.set_size_inches(10, 5 * len(images))
    for i, (imageA, imageB, imageC) in enumerate(images):
        imgAnp, imgBnp, imgCnp = (
            imageA.cpu().numpy().transpose((1, 2, 0)),
            imageB.cpu().numpy().transpose((1, 2, 0)),
            imageC.cpu().numpy().transpose((1, 2, 0)),
        )
        # compute the mean squared error and structural similarity
        # index for the images
        m = _mse(imgAnp, imgBnp)
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, 0))
        # show first image
        ax = fig.add_subplot(len(images), 3, i * 3 + 1)
        ax.set_title(f"Actual: {str(actuals[i].item())}")
        plt.imshow(imgAnp)
        plt.axis("off")
        # show the second image
        ax = fig.add_subplot(len(images), 3, i * 3 + 2)
        ax.set_title(f"Masked")
        plt.imshow(imgBnp)
        plt.axis("off")

        ax = fig.add_subplot(len(images), 3, i * 3 + 3)
        ax.set_title(f"Predicted: {str(predictions[i].item())}")
        plt.imshow(imgCnp)
        plt.axis("off")
    # show the images
    plt.show()
