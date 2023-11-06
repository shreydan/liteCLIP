# liteCLIP

## CLIP

CLIP (Contrastive Language-Image Pre-Training) model is a deep learning model designed to understand the relationship between images and text. Specifically, CLIP is trained on a large corpus of text and images in a self-supervised manner to learn how to associate descriptive text with the visual content of images.

#### It was introduced by [OpenAI](https://openai.com/research/clip)

#### Paper: Learning Transferable Visual Models From Natural Language Supervision [(arxiv)](https://arxiv.org/abs/2103.00020)

![contrastive pre-training](https://openaicom.imgix.net/fbc4f633-9ad4-4dc2-bd94-0b6f1feee22f/overview-a.svg?fm=auto&auto=compress,format&fit=min&w=3840&h=2733)

---

## liteCLIP

The models released were generally large in size since they used ViT and transformer language models as the image and text encoders respectively.

I wanted to train a lighter version of it to understand how it works and how the contrastive loss function associates the images with the texts so I trained liteCLIP.

I tried to implement the loss function as per the pseudo-code provided in the paper.

### trained using PyTorch, PyTorch Lightning

### it was trained on [Flickr8K](https://www.kaggle.com/datasets/adityajn105/flickr8k) which has ~8000 images with ~5 captions for each image.

### you can go through the training procedure in this notebook: [training.ipynb](./training.ipynb)

```
liteCLIP architecture:
----------------------

image encoder: convnext_tiny
text encoder: bert-mini (google/bert_uncased_L-4_H-256_A-4)
max token length: 128
embeddings dropout: 0.1
embeddings dimension: 256
batch size: 64
learning rate: 2e-4
epochs: 5
optimizer: Adam
```


## Zero-Shot Inference:

![zero-shot inference](https://openaicom.imgix.net/d9d46e4b-6d6a-4f9e-9345-5c6538b1b8c3/overview-b.svg?fm=auto&auto=compress,format&fit=min&w=3840&h=2946)

### Usage:

download model from `Releases`, save in `./model` dir as `liteclip2.pt`

```python
from liteclip import ZeroShotPipeline

pipeline = ZeroShotPipeline()

predictions = pipeline.predict('examples/cat.jpg',
                               ['a photo of a dog',
                                'a photo of a cat',
                                'the photo of a human baby'
                               ])

for label,prob in predictions:
    print(f"{label}: {prob*100:.2f}%")
```

### You can see the results in [inference.ipynb](./inference.ipynb)


### Extra Resources

- [@moein-shariatnia/OpenAI-CLIP](https://github.com/moein-shariatnia/OpenAI-CLIP)
- [@openai/CLIP](https://github.com/openai/CLIP/tree/main/clip)

## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.2103.00020,
  doi = {10.48550/ARXIV.2103.00020},
  url = {https://arxiv.org/abs/2103.00020},
  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Learning Transferable Visual Models From Natural Language Supervision},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

```bibtex
@article{turc2019,
  title={Well-Read Students Learn Better: On the Importance of Pre-training Compact Models},
  author={Turc, Iulia and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1908.08962v2 },
  year={2019}
}
```

```bibtex
@software{Shariatnia_Simple_CLIP_2021,
author = {Shariatnia, M. Moein},
doi = {10.5281/zenodo.6845731},
month = {4},
title = {{Simple CLIP}},
version = {1.0.0},
year = {2021}
}
```


had fun and learnt a lot <3