""" From this script the entire method can be executed, the train procedure as well as the test step.
    The parameters specified here are the ones related with remote sensing application.
"""
import os

Schedule = []
#Training the Domain Adaptation
Schedule.append("python train_remote_sensing.py --compute_ndvi False --buffer False " 
                "--name prove " 
                "--images_section Organized/Images/ --reference_section Organized/References/ "
                "--source_domain Amazon_PA --target_domain Cerrado_MA --stride_s 21 --stride_t 19 "
                "--source_image_name_T1 02_08_2016_image_R225_62 --source_image_name_T2 20_07_2017_image_R225_62 "
                "--target_image_name_T1 18_08_2017_image --target_image_name_T2 21_08_2018_image "
                "--dataroot /mnt/Data/Work/School/Trabajos_para_Tesis/Trabajo_Domain_Adaptation/Dataset/")
# Generating the translated domains
Schedule.append("python test_remote_sensing.py --model cycle_gan --compute_ndvi False --buffer False " 
                "--name prove " 
                "--images_section Organized/Images/ --reference_section Organized/References/ "
                "--source_domain Amazon_PA --target_domain Cerrado_MA --overlap_porcent_s 0.40 --overlap_porcent_t 0.40 "
                "--source_image_name_T1 02_08_2016_image_R225_62 --source_image_name_T2 20_07_2017_image_R225_62 "
                "--target_image_name_T1 18_08_2017_image --target_image_name_T2 21_08_2018_image "
                "--dataroot /mnt/Data/Work/School/Trabajos_para_Tesis/Trabajo_Domain_Adaptation/Dataset/")

for i in range(len(Schedule)):
    os.system(Schedule[i])
