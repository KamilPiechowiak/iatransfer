import os
from typing import Optional, Callable, Tuple, Any

from PIL import Image
from torchvision.datasets import VisionDataset


class Flowers102(VisionDataset):
    base_folder = 'flower_data'
    filename = 'flower_data.zip'

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        if download:
            self.download()
        super(Flowers102, self).__init__(root, transform=transform,
                                         target_transform=target_transform)

        assert (split in {'train', 'val', 'test'})
        if split == 'val':
            split = 'valid'
        downloaded_list = os.path.join(self.root, self.base_folder, split)

        self.data = []
        self.targets = []

        for i in range(102):
            for file_name in os.listdir(os.path.join(downloaded_list, str(i + 1))):
                if not file_name.endswith('.jpg'):
                    continue
                self.data.append(Image.open(os.path.join(downloaded_list, str(i + 1), file_name)))
                self.targets.append(i)

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
