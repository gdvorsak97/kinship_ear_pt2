from random import choice
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from kinship_utils import read_img, visualize_crop


class KinDatasetTest(Dataset):
    # may have to use load_data from datasets.py
    def __init__(self, test_dir, csv_file, transform=None):
        self.test_dir = test_dir
        self.results = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, index):
        items = self.results.img_pair.values[index]
        path1 = self.test_dir + items.split("g-")[0] + 'g'
        path2 = self.test_dir + items.split("g-")[1]
        label = self.results.ground_truth.values[index]

        """
        #cv2 -> pil
        img = cv2.imread("path/to/img.png")

        # You may need to convert the color.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        
        # For reversing the operation:
        im_np = np.asarray(im_pil)
        """

        # img1, img2 = Image.open(path1), Image.open(path2)
        img1 = read_img(path1)
        img2 = read_img(path2)

        # visualize_crop(img1, img2)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        # img1.show()

        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)  # if rescaling(tf) works, delete transform in main

        # img1.show()

        return img1, img2, label

    def __len__(self):
        return len(self.results)
