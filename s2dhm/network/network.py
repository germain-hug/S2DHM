"""Assemble image retrieval network, with intermediate endpoints.
"""
import gin
from collections import OrderedDict
from network.images_from_list import ImagesFromList
from network.netvlad import NetVLAD
from tqdm import tqdm
from typing import List
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.functional import interpolate
from torchvision import models

@gin.configurable
class ImageRetrievalModel():
    """Build the image retrieval model with intermediate feature extraction.

    The model is made of a VGG-16 backbone combined with a NetVLAD pooling
    layer.
    """
    def __init__(self, num_clusters: int, encoder_dim: int,
                checkpoint_path: str, hypercolumn_layers: List[int], device):
        """Initialize the Image Retrieval Network.

        Args:
            num_clusters: Number of NetVLAD clusters (should match pre-trained)
                weights.
            encoder_dim: NetVLAD encoder dimension.
            checkpoint_path: Path to the pre-trained weights.
            hypercolumn_layers: The hypercolumn layer indices used to compute
                the intermediate features.
            device: The pytorch device to run on.
        """
        self._num_clusters = num_clusters
        self._encoder_dim = encoder_dim
        self._checkpoint_path = checkpoint_path
        self._hypercolumn_layers = hypercolumn_layers
        self._device = device
        self._model = self._build_model()

    def _build_model(self):
        """ Build image retrieval network and load pre-trained weights.
        """
        model = nn.Module()

        # Assume a VGG-16 backbone
        encoder = models.vgg16(pretrained=False)
        layers = list(encoder.features.children())[:-2]
        encoder = nn.Sequential(*layers)
        model.add_module('encoder', encoder)

        # Assume a NetVLAD pooling layer
        net_vlad = NetVLAD(
            num_clusters=self._num_clusters, dim=self._encoder_dim)
        model.add_module('pool', net_vlad)

        # For parallel training
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            model.encoder = DataParallel(model.encoder)
            model.pool = DataParallel(model.pool)

        # Load weights
        checkpoint = torch.load(self._checkpoint_path,
                                map_location=lambda storage,
                                loc: storage)['state_dict']

        # If model was not trained in parallel, adapt the keys
        if 'module' not in list(checkpoint.keys())[0]:
            checkpoint = OrderedDict((k.replace('encoder', 'encoder.module'), v)
                for k, v in checkpoint.items())
            checkpoint = OrderedDict((k.replace('pool', 'pool.module'), v)
                for k, v in checkpoint.items())
        elif gpu_count <=1:
            checkpoint = OrderedDict((k.replace('encoder.module', 'encoder'), v)
                for k, v in checkpoint.items())
            checkpoint = OrderedDict((k.replace('pool.module', 'pool'), v)
                for k, v in checkpoint.items())


        model.load_state_dict(checkpoint)
        model = model.to(self._device)
        model.eval()
        return model

    @gin.configurable
    def compute_embedding(self, images, image_size, preserve_ratio):
        """Compute global image descriptor.

        Args:
            images: A list of image filenames.
            image_size: The size of the images to use.
            preserve_ratio: Whether the image ratio is preserved when resizing.
        Returns:
            descriptors: The global image descriptors, as numpy objects.
        """
        # Build dataloader
        dataloader = ImagesFromList(images, image_size,
                                    preserve_ratio=preserve_ratio)
        # Compute descriptors
        with torch.no_grad():
            db_desc = torch.zeros((len(dataloader),
                self._encoder_dim * self._num_clusters)).to(self._device)
            for i, tensor in tqdm(enumerate(dataloader), total=len(dataloader)):
                tensor = tensor.to(self._device).unsqueeze(0)
                db_desc[i,:] = self._model.pool(self._model.encoder(tensor))
        return db_desc.cpu().detach().numpy()

    @gin.configurable
    def compute_hypercolumn(self, image: List[str], image_size: List[int],
                            resize: bool, to_cpu: bool):
        """ Extract Multiple Layers and concatenate them as hypercolumns

        Args:
            image: A list of image paths.
            image_size: The maximum image size.
            resize: Whether images should be resized when loaded.
            to_cpu: Whether the resulting hypercolumns should be moved to cpu.
        Returns:
            hypercolumn: The extracted hypercolumn.
            image_resolution: The image resolution used as input.
        """
        # Pass list of image paths and compute descriptors
        with torch.no_grad():

            # Extract tensor from image
            feature_map = ImagesFromList.image_path_to_tensor(
                image_paths=image,
                image_size=image_size,
                resize=resize,
                device=self._device)
            image_resolution = feature_map[0].shape[1:]
            feature_maps, j = [], 0
            for i, layer in enumerate(list(self._model.encoder.children())):
                if(j==len(self._hypercolumn_layers)):
                    break
                if(i==self._hypercolumn_layers[j]):
                    feature_maps.append(feature_map)
                    j+=1
                feature_map = layer(feature_map)

            # Final descriptor size (concat. intermediate features)
            final_descriptor_size = sum([x.shape[1] for x in feature_maps])
            b, c, w, h = feature_maps[0].shape
            hypercolumn = torch.zeros(
                b, final_descriptor_size, w, h).to(self._device)

            # Upsample to the largest feature map size
            start_index = 0
            for j in range(len(self._hypercolumn_layers)):
                descriptor_size = feature_maps[j].shape[1]
                upsampled_map = interpolate(
                    feature_maps[j], size=(w, h),
                    mode='bilinear', align_corners=True)
                hypercolumn[:, start_index:start_index + descriptor_size, :, :] = upsampled_map
                start_index += descriptor_size

            # Delete and empty cache
            del feature_maps, feature_map, upsampled_map
            torch.cuda.empty_cache()

        # Normalize descriptors
        hypercolumn = hypercolumn / torch.norm(
            hypercolumn, p=2, dim=1, keepdim=True)
        if to_cpu:
            hypercolumn = hypercolumn.cpu().data.numpy()
        return hypercolumn, image_resolution

    @property
    def device(self):
        return self._device
