import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List
from PIL import Image, ImageFile
from torch.autograd import Variable
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesFromList(Dataset):
    """A generic data loader that loads images tensors from a list of filenames.
    """

    def __init__(self, images: List[str], image_size: List[int],
                 transform=None, loader=None, root=None, preserve_ratio=False):
        """Initialize ImagesFromList class.

        Args:
            images: A list of image filenames.
            image_size: Maximum size of the converted images.
            transform: Pytorch transform to be applied. If no transform is
                specified a default transform is applied.
            loader: The image loader to be used. If no loader is specified a
                default PIL loader is used.
            root: (Optional) An absolute path prefix for the image paths.
            preserve_ratio: Whether to preserve image ratio when resizing.
        """
        if len(images) == 0:
            raise(RuntimeError("Dataset contains 0 images!"))
        self._image_size = image_size
        self._images = images
        self._transform = transform
        self._loader = loader
        self._root = root
        self._preserve_ratio = preserve_ratio
        self._default_transform = self.default_transform()

    def __getitem__(self, index):
        """Open image and convert to pytorch tensor.
        """
        if self._root is not None:
            path = self._root + self._images[index]
        else:
            path = self._images[index]
        if self._loader is None:
            img = self.pil_loader(path)
        else:
            img = self._loader(path)
        if self._image_size is not None:
            img = self.imresize(img, self._image_size, self._preserve_ratio)
        if self._transform is not None:
            img = self._transform(img)
        else:
            img = self._default_transform(img)
        return img

    def default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def pil_loader(self, path: str):
        """Use PIL to open images.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def imresize(self, image, image_size: int, preserve_ratio=False):
        """Resize image, while optionally preserving image ratio.
        """
        if preserve_ratio:
            image.thumbnail((image_size, image_size), Image.ANTIALIAS)
            return image
        else:
            return image.resize((image_size, image_size), Image.ANTIALIAS)

    @classmethod
    def image_path_to_tensor(cls, image_paths: List[str], image_size: int,
                             resize: bool, device, transform=None):
        """Compute image tensor from a list of image paths.

        Args:
            image_paths: A list of image paths.
            image_size: Maximum image size.
            resize: Whether the images should be resized when opened.
            device: The pytorch device to use.
            transform: The transform to be applied. If no transform is specified
                a default transform is applied.
        Returns:
            images: The batch of image tensors.
            image_resolution: The resolution of the image tensors.
        """
        if resize:
            images = [cls.imresize(cls, image=Image.open(i), image_size=image_size,
                preserve_ratio=True) for i in image_paths]
        else:
            images = [Image.open(i) for i in image_paths]
        images = [cls.default_transform(cls)(i) for i in images]
        images = Variable(torch.stack(images).to(device), requires_grad=False)
        return images

    def __len__(self):
        return len(self._images)
