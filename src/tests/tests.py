import unittest
from dataset.lungpetctdx_dataset import LungPetCtDxDataset_TumorClass, LungPetCtDxDataset_TumorPresence
from dataset.ct_dataset import NormalizationMethods, normalize_meanstd, calculate_meanstd
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

class Test(unittest.TestCase):
    def test_normalization_whole_ds(self):
        # clear ds distribution cache
        os.remove("./data/LungPetCtDxDataset_dataset_distribution.pickle")

        print("Cleared ds distribution cache. Calculating distribution from scratch...")
        # Force load ds mean and std
        ds2 = LungPetCtDxDataset_TumorClass(normalize=NormalizationMethods.WHOLE_DATASET)

        print("Calculated ds distribution. Calculating distribution of normalized dataset (should be a standard normal distr.)...")
        # Calculate mean and sd of normalized dataset
        means, sds = ds2.calculateMeanAndSD(normalize=True)

        # Ensure we have a standard normal distribution with mean about 0 and sd about 1
        for m in means.tolist():
            self.assertAlmostEqual(m, 0, 5)
        for s in sds.tolist():
            self.assertAlmostEqual(s, 1, 5)

    def test_normalization_single_img(self):
        ds = LungPetCtDxDataset_TumorPresence(
            normalize=NormalizationMethods.SINGLE_IMAGE)

        loader = DataLoader(
            ds, batch_size=32, shuffle=True, num_workers=4)
        for idx, (inputs, _, _) in enumerate(loader):
            print(f"Done with {idx}/{len(loader)} batches.\r")

            for img in inputs:
                img = img.numpy().astype(np.float64)

                means, stds = calculate_meanstd(img, (1, 2))
                means, stds = np.squeeze(means), np.squeeze(stds)
                for m in means:
                    self.assertTrue(-.1 < m < .1)
                for s in stds:
                    self.assertTrue(0.9 < s < 1.1)

    def test_normalization_single_img_dummy(self):
        img = np.array(
            [[[1, 11], [2, 22]], [[3, 33], [4, 44]], [[5, 55], [6, 66]]])

        img = normalize_meanstd(img, (1, 2))
        means, stds = calculate_meanstd(img, (1, 2))
        means, stds = np.squeeze(means), np.squeeze(stds)
        print("means", means)
        print("stds", stds)
        for m in means:
            self.assertAlmostEqual(m, 0, 5)
        for s in stds:
            self.assertAlmostEqual(s, 1, 5)


if __name__ == "__main__":
    unittest.main()
