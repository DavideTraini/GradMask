# Grad-Mask

This is the official implementation of the paper: Integrating Gradient and Mask-based Approaches for Vision Transformer Explainability.

## Abstract

Vision Transformers (ViTs) have demonstrated outstanding performance across different computer vision tasks thanks to their self-attention mechanism that captures long-range dependencies effectively. However, the inherent complexity of ViTs presents significant challenges in explaining their outputs, which is fundamental in safety-critical domains. To tackle the challenge of explaining ViT outputs, this paper presents GradMask, a novel method that integrates gradients into the mask generation process to create explanation heatmaps. GradMask uses the query, key, and value matrices from each attention layer and computes their gradients with respect to a target class. Afterward, it uses these gradients to generate binary masks, which are then weighted by the corresponding ViT's confidence scores. Finally, it combines the weighted masks to generate the resulting heatmap. Experimental evaluations on an ImageNet subset with ViT and DeiT (Data-efficient Image Transformer) architectures show that GradMask achieves competitive performance according to standard explainability metrics, such as Insertion, Deletion, and Pointing Game. A hyperparameter analysis confirms the high computational efficiency of GradMask, while an ablation study highlights the importance of combining gradients and masks for the generation of the explanation heatmap. Finally, a qualitative analysis shows the improved explainability of GradMask compared to existing methods, making it a promising approach for understanding ViTs.


![Comparison-Methods](./Readme_images/Compare.jpg)



## Usage

This repository can be directly downloaded and executed locally. The required libraries are displayed in Section [Requirements](#requirements)

In the **Implementation** folder we provide a file called **Usage_example** where the user can run the cells and visualize insertion AUC, deletion AUC and heatmaps of one or more images. 
The code can be used both for ViT and DeiT explaination. In addition, this folder contains the following python files:
- **utils**: contains some functions used for the visualization of AUCs and heatmaps.
- **feature_extractor**: contains the definition of the feature extractor class used by ViT and DeiT models.
- **hook**: contains the definition of the hook classes used by ViT and DeiT models.
- **GradMask**: contains the implementation of the model described in the paper.

The **imgs_idx** file contains the indexes of the images used for testing our approach, which are selected from the ImageNet 2012 validation dataset. To download the dataset you need to login on [ImageNet site](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and click on ILSVRC2012, after this download the 'Development kit (Task 1 & 2)', that contains the ground truth labels, and the 'Validation images (all tasks)'.


## Parameters 
In the **Usage_sample** file the user can modify the following parameters:

### Initialization parameters
**`model`**: represents the model of which we want to make explainability; in this demo this value can be _'vit'_ or _'deit'_;  
**`device`**: device in which the model will be used; this can be _'cpu'_ or _'cuda:0'_;  

### Call Parameters
**`masks_layer`**: number of masks for every layer;  
**`img_path`**: path of the image we want to explain;  
**`label`**: label on which we want to do explaination; this value is optional and if is not provided the model will give the heatmap associated with the predicted class;  




## Requirements <a name="requirements"></a>

In our notebook we used the following libraries:
```
PIL=10.3.0  
transformers=4.30.2
torch=2.2.2
torchvision=0.17.2
numpy=1.26.4
pandas=2.2.2
matplotlib=3.9.0
seaborn=0.13.2
```

## Citation

If you use this model for your research please cite our paper.
