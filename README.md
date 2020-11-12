# CycleGAN-based Domain Adaptation for Deforestation Detection.

The current project contains the scripts to perform a Domain Adaptation based on CycleGAN technique for change detection in remote sensing data, specifically for deforestation detection in two Brazilian biomes, the Amazon rainforest(Brazilian Legal Amazon) and Brazilian savannah (Cerrado).

The following figure shows overall architecture of the proposed methodology. As in CycleGAN framework, the model contains two mapping functions $G$ and as well as the associated discriminators and. encourages the to translate into outcomes indistinguishable of , and vice versa for and .  To further regularizes the translation procedure, weintroduced twodifference losseswhich aim at preserving the changes between the images of both domains in their respectivetranslated versions. Although not represented, the model also uses the already knowncycle consistency lossand theidentity lossdefined in [1].

![Image](Methodology.jpg)

# References
