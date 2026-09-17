import logging
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
logger.setLevel(logging.ERROR)
import os
import pandas as pd
import numpy as np
from radiomics import featureextractor
from natsort import ns, natsorted
import SimpleITK as sitk
import scipy.ndimage as ndimage
from radiomics import featureextractor
from skimage import morphology

import warnings
warnings.filterwarnings('ignore')


'''FeatureExtract-original mask'''
params_path = os.path.abspath('FeatureExtractParams.yaml')
extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
#table
table = pd.DataFrame()
# original data and mask
folder_path1 = '/imageFolder'
folder_path2 = '/maskFolder'
#savePath
out_path = '/savePath'
output_path = os.path.join(out_path, 'Extracted_Feature_original.csv')
folder_list = os.listdir(folder_path2)
for folder in folder_list:
    try:
        print(folder)
        subfolder_path = os.path.join(folder_path2,folder)
        subfolder_list = os.listdir(subfolder_path)
        for subfolder in subfolder_list:
            name_path = os.path.join(subfolder_path,subfolder)
            mask_path = os.path.join(name_path,'mask.nii')
            ct_path = os.path.join(os.path.join(folder_path1, folder,subfolder),'image.nii')
            featureVector = pd.Series(extractor.execute(ct_path, mask_path))
            featureVector.name = str(subfolder)
            table = table.join(featureVector, how='outer')
    except:
        print(folder,'Problem!')
table = table.T
drop_dp = table.filter(regex=('diagnostics.*'))
table = table.drop(drop_dp.columns, axis=1)
table.to_csv(output_path)



'''FeatureExtract-peri mask'''
params_path = os.path.abspath('FeatureExtractParams.yaml')
extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
#table
table = pd.DataFrame()
# original data and mask
folder_path1 = '/imageFolder'
folder_path2 = '/maskFolder'
mask_out_path = '/periMaskFolder'
#savePath
out_path = '/savePath'
output_path = os.path.join(out_path, 'Extracted_Feature_peri5.csv')
folder_list = os.listdir(folder_path2)
for folder in folder_list:
    try:
        print(folder)
        subfolder_path = os.path.join(folder_path2,folder)
        subfolder_list = os.listdir(subfolder_path)
        for subfolder in subfolder_list:
            name_path = os.path.join(subfolder_path,subfolder)
            mask_path = os.path.join(name_path,'mask.nii')
            ct_path = os.path.join(os.path.join(folder_path1, folder,subfolder),'image.nii')
            mask_image = sitk.ReadImage(mask_path)
            # Calculate radius in voxels (2.5 mm outward and 2.5 mm inward)
            spacing = np.array(mask_image.GetSpacing())   ###0.5*0.5*0.5 spacing
            expansion_pixels = int(np.round(2.5 / spacing[0]))
            # Ensure binary mask array
            mask_array = (sitk.GetArrayFromImage(mask_image) > 0).astype(np.uint8)
            # Binary dilation and erosion
            selem = morphology.ball(expansion_pixels)
            dilated_mask_array = morphology.binary_dilation(mask_array, selem)
            eroded_mask_array = morphology.binary_erosion(mask_array, selem)
            # Difference between dilated and eroded masks
            edge_region_mask_array = dilated_mask_array ^ eroded_mask_array
            #  new mask
            edge_region_mask = sitk.GetImageFromArray( edge_region_mask_array.astype(np.uint8) )
            edge_region_mask.CopyInformation(mask_image)
            if not os.path.exists(os.path.join(mask_out_path,folder,subfolder)):
                os.makedirs(os.path.join(mask_out_path,folder,subfolder))
            sitk.WriteImage(edge_region_mask, os.path.join(mask_out_path,folder,subfolder,'mask.nii')) ####need check
            ###==========after check=========
            new_mask_path = os.path.join(mask_out_path,folder,subfolder,'mask.nii')
            #####****************************************************************
            ct_path = os.path.join(os.path.join(folder_path1, folder,subfolder),'image.nii')
            featureVector = pd.Series(extractor.execute(ct_path, new_mask_path))
            featureVector.name = str(folder+'-'+subfolder)
            table = table.join(featureVector, how='outer')
    except:
        print(folder,'Problem!')
table = table.T
drop_dp = table.filter(regex=('diagnostics.*'))  # 删掉提示信息列
table = table.drop(drop_dp.columns, axis=1)
table.to_csv(output_path)