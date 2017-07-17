# Super-Resolution Generative Adversarial Networks (SRGAN)
## About this project
This project aims to implement SRGAN based on the Christian Ledig et al's "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (See: https://arxiv.org/abs/1609.04802) with Tensorflow.  

In this project, we might incorporate the contents in Martin Arjovsky et al's "Wasserstein GAN" (See: https://arxiv.org/abs/1701.07875) and Huikai Wu el al's "GP-GAN_Towards Realistic High-Resolution Image Blending" (See: https://arxiv.org/abs/1703.07195) to achieve better performance or so.  

## Current Progress  
Implementing VGG19 (pending for verification)  
Implementing SRGAN (revising loss function from https://github.com/tadax/srgan )  

## Project Roadmap  
1. Implement VGG19  
2. Verify VGG19  
3. Implement SRGAN  
4. Verify SRGAN  
5. Train VGG19  
6. Train SRGAN  
7. Evaluate performance on SRGAN  

## Issues  
\#1. How to train VGG19 with Super-Resolution purpose?  

## References
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network  
Christian Ledig et al  
https://arxiv.org/abs/1609.04802  
  
Wasserstein GAN  
Martin Arjovsky, Soumith Chintala, Léon Bottou  
https://arxiv.org/abs/1701.07875  
  
GP-GAN: Towards Realistic High-Resolution Image Blending  
Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang  
https://arxiv.org/abs/1703.07195  
  
How to Train Your DRAGAN  
Naveen Kodali, Jacob Abernethy, James Hays, Zsolt Kira  
https://arxiv.org/abs/1705.07215  

Deep Residual Learning for Image Recognition  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
https://arxiv.org/abs/1512.03385  