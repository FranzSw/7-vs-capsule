import unittest
from lungpetctdx_dataset import LungPetCtDxDataset
import os


class Test(unittest.TestCase):
    def test_normalization(self):
        # clear ds distribution cache
        os.remove("./data/LungPetCtDxDataset_dataset_distribution.pickle")

        print("Cleared ds distribution cache. Calculating distribution from scratch...")
        # Force load ds mean and std
        ds2 = LungPetCtDxDataset(normalize=True)

        print("Calculated ds distribution. Calculating distribution of normalized dataset (should be a standard normal distr.)...")
        # Calculate mean and sd of normalized dataset
        means, sds = ds2.calculateMeanAndSD(normalize=True)

        # Ensure we have a standard normal distribution with mean about 0 and sd about 1
        for m in means.tolist():
            self.assertAlmostEqual(m, 0, 5)
        for s in sds.tolist():
            self.assertAlmostEqual(s, 1, 5)


if __name__ == "__main__":
    unittest.main()
