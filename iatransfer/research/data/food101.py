import copy
import os
from typing import Optional, Callable, Tuple, Any

from PIL import Image
from torchvision.datasets import VisionDataset
from tqdm import tqdm


class Food101(VisionDataset):
    base_folder = 'food-101'
    filename = 'food-101.tar.gz'

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(Food101, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        if download:
            self.download()

        self.train = train  # training set or test set
        if train:
            downloaded_list = os.path.join(self.root, self.base_folder, 'meta', 'train.txt')
        else:
            downloaded_list = os.path.join(self.root, self.base_folder, 'meta', 'test.txt')

        classes = {}
        with open(os.path.join(self.root, self.base_folder, 'meta', 'classes.txt')) as f:
            for i, clazz in enumerate(f.read().split('\n')):
                classes[clazz] = i

        self.data = []
        self.targets = []

        with open(downloaded_list) as f:
            for file_name in tqdm(f.read().split('\n')[:-1]):
                self.data.append(
                    copy.deepcopy(Image.open(os.path.join(self.root, self.base_folder, 'images', f'{file_name}.jpg'))))
                self.targets.append(classes[file_name.split('/')[0]])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        '''
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        '''
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        pass
