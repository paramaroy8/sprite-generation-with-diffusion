## Summary

In this project, image-to-image generation has been implemented using diffusion model. The U-net architecture with context and time embeddings was used as the diffusion model. 16-by-16 Sprite images were used to train and evaluate the model. 

## Why Sprite Images Instead Of Real-life Images?

Sprite images have reduced complexities compared to real-life images. As a result, this dataset is less computationally expensive, making it a efficient for fast experimentation with different types of models and techniques.

## Context Embeddings

Including context embeddings help users to have better control in the diffusion process.
While text embeddings can serve as contexts, different categories of images can also serve as contexts.
In this project, I used 5 contexts.

Contexts can be randomly generated, user defined or mixed.

## Denoising Algorithm

The denoising process implemented in this project is based on DDPM algorithm. Random noises were sampled, and then predicted noise were removed over timesteps to generate new images.

## Acknowledgments

Dataset: [Sprite Images Dataset](https://huggingface.co/datasets/ashis-palai/sprites_image_dataset/tree/main)

Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

Tutorial: [How Diffusion Models Work](https://learn.deeplearning.ai/courses/diffusion-models/lesson/xb8aa/introduction)

GitHub Repo: [minDiffusion](https://github.com/cloneofsimo/minDiffusion)
