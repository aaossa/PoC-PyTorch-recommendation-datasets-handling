# PyTorch recommendation datasets handling

###### (Proof of concept)

This is part of the software support I need in order to complete my master thesis in time. The idea is to provide a minimal standard for handling datasets for recommender systems, in which the items are images (or embeddings).

## Requirements

What I need is to handle large amounts of memory, mostly because of the images. I have access to GPUs, so including options to handle these objects directly in CUDA would be nice. The data should be ordered to easily access to images and user history. Also, I need to calculate multiple ranking metrics, so I need to access the user history in a fast manner too.

I'm working with models that are trained using a BPR setting, with possitive and negative items, but PyTorch `Dataset`s/`DataLoader`s do not include this option (training using triplets) directly. I would need to retrieve multiple images fast and pass them while the model is training.

## Inspiration from

* [PyTorch `Dataset`](https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py): This is the base class I'll be using, so I need to understand its basics to work with the next item.
* [`torchvision` datasets](https://github.com/pytorch/vision/tree/master/torchvision/datasets): These already handle images and include a common base. The problem is that they're not specialized for recommendation tasks, but the main structure will be useful (such as downloading the data and support for `torchvision.transforms`).
* [`cornac` `Dataset`](https://github.com/PreferredAI/cornac/blob/master/cornac/data/dataset.py): This library includes multiple some image [datasets](https://github.com/PreferredAI/cornac/tree/master/cornac/datasets) and is really nice to work with, but is not based in PyTorch. I like the internals of it (such as the usage of `scipy.sparse`), but I'll need to make it work with PyTorch.

## Initial ideas

* Download data as a PyTorch `Dataset` does
* Store and handle images as `torchvision` does
* Store and handle interactions as `cornac` does

## Further questions

* How to store and load embeddings efficiently?
* Is it possible to connect the pipeline to [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)?
* Where to include the dataset split procedure?

