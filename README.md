# CycleGAN-based Domain Adaptation for Deforestation Detection.

The current project contains the scripts to perform a Domain Adaptation based on CycleGAN technique for change detection in remote sensing data, specifically for deforestation detection in two Brazilian biomes, the Amazon rainforest(Brazilian Legal Amazon) and Brazilian savannah (Cerrado).

The following figure shows overall architecture of the proposed methodology. As in CycleGAN framework, the model contains two mapping functions G:X->Y and F:Y->X as well as the associated discriminators Dx and Dy. Dx encourages the generator F to translate Y into outcomes indistinguishable of X, and vice versa for Dy and Y. To further regularizes the translation procedure, we introduced two difference losses which aim at preserving the changes between the images of both domains in their respective translated versions. Although not represented, the model also uses the already known cycle consistency loss and the identity loss defined in [1].

![Image](Methodology.jpg)

# References
