from random import choice

from PIL import Image
from torch.utils.data import Dataset


class KinDataset(Dataset):
    # TODO: make sure data is organised correctly or try to use the same methods as in kinship
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

        # TODO: switch the image reading with the method from main repo,
        #  put the methods in a utils.py file and import them here
        img1, img2 = Image.open(path1), Image.open(path2)

        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.relations)*2
