# CycleGAN-based Domain Adaptation for Deforestation Detection.

The current project contains the scripts to perform a Domain Adaptation based on CycleGAN technique for change detection in remote sensing data, specifically for deforestation detection in two Brazilian biomes, the Amazon rainforest(Brazilian Legal Amazon) and Brazilian savannah (Cerrado).

The following figure shows overall architecture of the proposed methodology. As in CycleGAN framework, the model contains two mapping functions G:X->Y and F:Y->X as well as the associated discriminators Dx and Dy. Dx encourages the generator F to translate Y into outcomes indistinguishable of X, and vice versa for Dy and Y. To further regularizes the translation procedure, we introduced two difference losses which aim at preserving the changes between the images of both domains in their respective translated versions. Although not represented, the model also uses the already known cycle consistency loss and the identity loss defined in [1].

![Image](Methodology.jpg)

# Prerequisites
1- Python 3.7.4

2- Pytorch 1.5

3- Tensorflow 2

4- [CycleGAN original code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) prerequisites. 

# The framework

This implementation is an adaptation of the [CycleGAN Pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for remote sensing data(Landsat 8 OLI). The main noveltie in this release is the introduction of a new constraint into the model's cost function to alleviate the artifacts generation as well as the adaptation to work with remote sensing images of different dimensions.

# Train and Test

python 

# References

[1] J. Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks,” Proc. IEEE Int. Conf. Comput. Vis., vol. 2017-Octob, pp. 2242–2251, 2017, doi: 10.1109/ICCV.2017.244.
