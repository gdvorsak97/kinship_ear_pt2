from random import choice

import cv2
from PIL import Image
from torch.utils.data import Dataset

from kinship_utils import read_img


class KinDataset(Dataset):
    # may have to use load_data from datasets.py
    def __init__(self, relations, person_to_images_map, transform=None):
        self.relations = relations
        self.transform = transform
        self.person_to_images_map = person_to_images_map
        self.people = list(person_to_images_map.keys())

    def __getitem__(self, index):
        if index % 2 == 0:  # Positive samples
            p1, p2 = self.relations[index // 2]
            label = 1
        else:  # Negative samples
            while True:
                p1 = choice(self.people)
                p2 = choice(self.people)
                if p1 != p2 and (p1, p2) not in self.relations and (p2, p1) not in self.relations:
                    break
            label = 0

        path1, path2 = choice(self.person_to_images_map[p1]), choice(self.person_to_images_map[p2])

        """
        #cv2 -> pil
        img = cv2.imread("path/to/img.png")

        # You may need to convert the color.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        
        # For reversing the operation:
        im_np = np.asarray(im_pil)
        """

        # TODO: test color conversion and compare transforms
        # img1, img2 = Image.open(path1), Image.open(path2)
        img1 = read_img(path1)
        img2 = read_img(path2)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # test - visualize, can use visualize_crop
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)  # if rescaling(tf) works, delete transform in main

        return img1, img2, label

    def __len__(self):
        return len(self.relations)*2
