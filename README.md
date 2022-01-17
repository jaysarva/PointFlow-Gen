# PointFlow -- Point Cloud Generation
![Blueno collage](/generator_outputs/collage/collage.png)
This is my implementation of PointFlow, a generative model proposed by Yang et al. (2019). 

## Running the model
To train and test your model, run the respective files. I've included one set of checkpoints to visualize on your own. Make sure that you install [torchdiffeq](https://github.com/rtqichen/torchdiffeq) before running. 

## Basics about the Model
This model essentially learns two sets of probability distributions where one is the distribution of 'shapes' and the second level is the distribution of specific points given a shape that is being modeled. 

## Dataset
I trained on airplane meshes from the ShapeNetCore dataset, using the PyTorch3D data loader. 

## Results
It should be noted that the below results are not particularly well trained due to Google Colab not being very cooperative, leading me to have to train locally. With many more epochs, results may be substantially better. 

Below are some generated images after training. Point clouds were rendered with [this tool](https://github.com/zekunhao1995/PointFlowRenderer)

![](/imgs/im1.png)
![](/imgs/im2.png)



## References
[1] Ruihui Li, Xianzhi Li, Ka-Hei Hui, and Chi-Wing Fu. 2021. SP-GAN: Sphere-Guided 3D Shape Generation and Manipulation. ACM Trans. Graph. 40, 4,
Article 151 (August 2021), 13 pages. https://doi.org/10.1145/3450626.3459766
