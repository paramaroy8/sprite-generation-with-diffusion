Project Description
In this project, image-to-image generation has been implemented using diffusion model. The U-net architecture with context and time embeddings was used as the diffusion model. 16-by-16 Sprite images were used to train and evaluate the model.

Why Sprite Images Instead Of Real-life Images?
Sprite images have reduced complexities compared to real-life images. As a result, this dataset is less computationally expensive, making it a efficient for fast experimentation with different types of models and techniques.

Context Embeddings
Including context embeddings help users to have better control in the diffusion process. While text embeddings can serve as contexts, different categories of images can also serve as contexts. In this project, I used 5 contexts.

Contexts can be randomly generated, user defined or mixed.

Acknowledgments
Dataset: Sprite Images Dataset

Paper: Denoising Diffusion Probabilistic Models

Tutorial: How Diffusion Models Work

GitHub Repo: minDiffusion
