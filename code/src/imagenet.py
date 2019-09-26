import os

from nltk.corpus import wordnet as wn
import torch

import random

from skimage import io
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from contextlib import redirect_stdout, redirect_stderr
import io as py_io

IMAGENET = 'data/imagenet'

MAX_IMAGES_PER_SYNSET = 256

class VisualEmbeddings:
    def __init__(self, imagenet_dir=IMAGENET):
        self.sampler = ImageNetSampler(imagenet_dir=imagenet_dir)
        self.vgg = models.vgg19(pretrained=True).cuda()

    # Return shape: [batch, features]
    def _classifier_embeddings(self, batch):
        '''Embedding obtained from first ReLU layer of VGG.'''

        self.vgg.train(False)
        curr = batch
        curr = self.vgg.features(curr)
        curr = curr.view(len(curr), -1)

        for layer in self.vgg.classifier:
            curr = layer(curr)
            if isinstance(layer, nn.ReLU):
                return curr

        assert(False)

    def embedding_and_imagenet_paths_for_synset(self, synset):
        if isinstance(synset, str):
            synset = wn.synset(synset)

        try:
            sample = self.sampler.sample_images_for_synset(
                    synset, MAX_IMAGES_PER_SYNSET)
        except FileNotFoundError:
            sample = []

        if not len(sample):
            print("Warning: no images for {}".format(synset))
            return None, []

        print("Computing embedding for %s" % synset.name())

        dataset    = ImageNetDataset(sample)
        dataloader = DataLoader(dataset, batch_size=32)

        total     = 0
        total_num = 0
        for batch in dataloader:
            embs = self._classifier_embeddings(batch)
            embs = embs.detach()
            total     += torch.sum(embs, dim=0)
            total_num += len(embs)

        unnorm = total / total_num
        embedding = unnorm / torch.norm(unnorm)

        return embedding, sample

class ImageNetSampler():
    '''
    Class that handles ImageNet and its related WordNet nodes and
    image samples.
    Assumes that all words are nouns.
    '''

    def __init__(self, imagenet_dir):
        self.dir = imagenet_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                # Parameters from PyTorch docs for pretrained torchvision models.
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])

    def _is_readable_3_channels_img(self, path):
        try:
            f = py_io.StringIO()
            with redirect_stdout(f):
                with redirect_stderr(f):
                    img = io.imread(path)
                    self.transform(img)
            if "UserWarning" in f.getvalue():
                return False
            return (len(img.shape) == 3 and img.shape[2] == 3)
        except:
            return False

    def sample_images_for_synset(self, synset, num):
        '''
        Given a WordNet synset, produce `num` (or as many as possible) images
        from the ImageNet database that can be successfully read and processed
        by PyTorch.
        Images are sampled randomly.
        Returns paths to images.
        '''

        img_dir   = os.path.join(self.dir, self._dir_name(synset.offset()))
        all_files = os.listdir(img_dir)
        random.shuffle(all_files)

        res = []
        for file in all_files:
            if len(res) >= num: break

            path = os.path.join(img_dir, file)
            if self._is_readable_3_channels_img(path):
                res.append(path)

        return res

    def _dir_name(self, id):
        return 'n%08d' % id

class ImageNetDataset(Dataset):
    '''
    A PyTorch Dataset for images sampled by the ImageNetSampler class.
    '''

    def __init__(self, paths):
        '''
        Init the dataset to process all images given by the list `paths`.
        '''

        self.paths = paths
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                # Parameters from PyTorch docs for pretrained torchvision models.
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = io.imread(path)
        return self.transform(img).cuda()
