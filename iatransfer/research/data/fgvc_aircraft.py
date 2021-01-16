import copy
import os
from typing import Optional, Callable, Tuple, Any

from PIL import Image
from torchvision.datasets import VisionDataset
from tqdm import tqdm


class FGVCAircraft(VisionDataset):
    base_folder = 'fgvc-aircraft-2013b'
    filename = 'fgvc-aircraft-2013b.tar.gz'

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(FGVCAircraft, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        if download:
            self.download()

        assert (split in {'train', 'val', 'test'})
        if split == 'train':
            split = 'trainval'
        downloaded_list = os.path.join(self.root, self.base_folder, 'data', f'images_variant_{split}.txt')

        classes = {}
        with open(os.path.join(self.root, self.base_folder, 'data', 'variants.txt')) as f:
            for i, clazz in enumerate(f.read().split('\n')):
                classes[clazz] = i

        self.data = []
        self.targets = []

        with open(downloaded_list) as f:
            for row in tqdm(f.read().split('\n')[:-1]):
                file_name = row.split(' ')[0]
                variant = ' '.join(row.split(' ')[1:])
                self.data.append(copy.deepcopy(
                    Image.open(os.path.join(self.root, self.base_folder, 'data', 'images', f'{file_name}.jpg'))))
                self.targets.append(classes[variant])

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
