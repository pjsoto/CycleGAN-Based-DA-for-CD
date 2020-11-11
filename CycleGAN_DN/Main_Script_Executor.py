""" From this script the entire method can be executed, the train procedure as well as the test step.
    The parameters specified here are the ones related with remote sensing application.
"""
import os

Schedule = []
#Training the Domain Adaptation
Schedule.append("python train_remote_sensing.py --compute_ndvi False --buffer False "  
                "--name Prove " 
                "--images_section Organized/Images/ --reference_section Organized/References/ "
                "--source_domain Amazon_RO --target_domain Amazon_PA --stride_s 50 --stride_t 19 "
                "--source_image_name_T1 18_07_2016_image  --source_image_name_T2 21_07_2017_image "
                "--target_image_name_T1 02_08_2016_image_R225_62 --target_image_name_T2 20_07_2017_image_R225_62 "
                "--dataroot /mnt/Data/Work/School/Trabajos_para_Tesis/Trabajo_Domain_Adaptation/Dataset/")
# Generating the translated domains
Schedule.append("python test_remote_sensing.py --model cycle_gan --compute_ndvi False --buffer False " 
                "--name Prove " 
                "--images_section Organized/Images/ --reference_section Organized/References/ "
                "--source_domain Amazon_RO --target_domain Amazon_PA --overlap_porcent_s 0.42 --overlap_porcent_t 0.75 "
                "--source_image_name_T1 18_07_2016_image --source_image_name_T2 21_07_2017_image "
                "--target_image_name_T1 02_08_2016_image_R225_62 --target_image_name_T2 20_07_2017_image_R225_62 "
                "--dataroot /mnt/Data/Work/School/Trabajos_para_Tesis/Trabajo_Domain_Adaptation/Dataset/")

for i in range(len(Schedule)):
    os.system(Schedule[i])
