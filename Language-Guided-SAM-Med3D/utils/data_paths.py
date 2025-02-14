img_datas = [
'/home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/gland_train',
# '/home/haojing/workspace/MICCAI25/MRI_data/gland_test',
]
img_datas_val = [
'/home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/gland_test',
]
all_classes = [
'COVID_lesion',
'adrenal',
'adrenal_gland_left',
'adrenal_gland_right',
'airway',
'aorta',
'autochthon_left',
'autochthon_right',
'bilateral_optic_nerves',
'bilateral_parotid_glands',
'bilateral_submandibular_glands',
'bladder',
'bone',
'brain',
'brain_lesion',
'brainstem',
'buckle_rib_fracture',
'caudate_left',
'caudate_right',
'cerebellum',
'cerebral_microbleed',
'cerebrospinal_fluid',
'clavicula_left',
'clavicula_right',
'cocygis',
'colon',
'colon_cancer_primaries',
'deep_gray_matter',
'displaced_rib_fracture',
'duodenum',
'edema',
'enhancing_tumor',
'esophagus',
'external_cerebrospinal_fluid',
'face',
'femur_left',
'femur_right',
'gallbladder',
'gluteus_maximus_left',
'gluteus_maximus_right',
'gluteus_medius_left',
'gluteus_medius_right',
'gluteus_minimus_left',
'gluteus_minimus_right',
'gray_matter',
'head_of_femur_left',
'head_of_femur_right',
'heart',
'heart_ascending_aorta',
'heart_atrium_left',
'heart_atrium_left_scars',
'heart_atrium_right',
'heart_blood_pool',
'heart_left_atrium_blood_cavity',
'heart_left_ventricle_blood_cavity',
'heart_left_ventricular_myocardium',
'heart_myocardium',
'heart_myocardium_left',
'heart_right_atrium_blood_cavity',
'heart_right_ventricle_blood_cavity',
'heart_ventricle_left',
'heart_ventricle_right',
'hepatic_tumor',
'hepatic_vessels',
'hip_left',
'hip_right',
'hippocampus_anterior',
'hippocampus_posterior',
'humerus_left',
'humerus_right',
'iliac_artery_left',
'iliac_artery_right',
'iliac_vena_left',
'iliac_vena_right',
'iliopsoas_left',
'iliopsoas_right',
'inferior_vena_cava',
'intestine',
'ischemic_stroke_lesion',
'kidney',
'kidney_cyst',
'kidney_left',
'kidney_right',
'kidney_tumor',
'left_eye',
'left_inner_ear',
'left_lens',
'left_mandible',
'left_middle_ear',
'left_optical_nerve',
'left_parotid_gland',
'left_temporal_lobes',
'left_temporomandibular_joint',
'left_ventricular_blood_pool',
'left_ventricular_myocardial_edema',
'left_ventricular_myocardial_scars',
'left_ventricular_normal_myocardium',
'liver',
'liver_tumor',
'lumbar_vertebra',
'lung',
'lung_cancer',
'lung_infections',
'lung_left',
'lung_lower_lobe_left',
'lung_lower_lobe_right',
'lung_middle_lobe_right',
'lung_node',
'lung_right',
'lung_upper_lobe_left',
'lung_upper_lobe_right',
'lung_vessel',
'mandible',
'matter_tracts',
'multiple_sclerosis_lesion',
'myocardial_infarction',
'nasopharynx_cancer',
'no_reflow',
'non_displaced_rib_fracture',
'non_enhancing_tumor',
'optic_chiasm',
'other_pathology',
'pancreas',
'pancreatic_tumor_mass',
'pituitary',
'portal_vein_and_splenic_vein',
'prostate',
'prostate_and_uterus',
'prostate_peripheral_zone',
'prostate_transition_zone',
'pulmonary_artery',
'rectum',
'renal_artery',
'renal_vein',
'rib_left_1',
'rib_left_10',
'rib_left_11',
'rib_left_12',
'rib_left_2',
'rib_left_3',
'rib_left_4',
'rib_left_5',
'rib_left_6',
'rib_left_7',
'rib_left_8',
'rib_left_9',
'rib_right_1',
'rib_right_10',
'rib_right_11',
'rib_right_12',
'rib_right_2',
'rib_right_3',
'rib_right_4',
'rib_right_5',
'rib_right_6',
'rib_right_7',
'rib_right_8',
'rib_right_9',
'right_eye',
'right_inner_ear',
'right_lens',
'right_mandible',
'right_middle_ear',
'right_optical_nerve',
'right_parotid_gland',
'right_temporal_lobes',
'right_temporomandibular_joint',
'right_ventricular_blood_pool',
'sacrum',
'scapula_left',
'scapula_right',
'segmental_rib_fracture',
'small_bowel',
'spinal_cord',
'spleen',
'stomach',
'trachea',
'unidentified_rib_fracture',
'urinary_bladder',
'uterus',
'ventricles',
'vertebrae_C1',
'vertebrae_C2',
'vertebrae_C3',
'vertebrae_C4',
'vertebrae_C5',
'vertebrae_C6',
'vertebrae_C7',
'vertebrae_L1',
'vertebrae_L2',
'vertebrae_L3',
'vertebrae_L4',
'vertebrae_L5',
'vertebrae_L6',
'vertebrae_T1',
'vertebrae_T10',
'vertebrae_T11',
'vertebrae_T12',
'vertebrae_T13',
'vertebrae_T2',
'vertebrae_T3',
'vertebrae_T4',
'vertebrae_T5',
'vertebrae_T6',
'vertebrae_T7',
'vertebrae_T8',
'vertebrae_T9',
'white_matter',
'white_matter_hyperintensity',
]

all_datasets = [
'AMOS2022_ct',
'AMOS2022_mr_unknown',
'ATLAS2_mr_t1w',
'ATM2022_ct',
'AbdomenCT1K_ct',
'BTCV_Abdomen_ct',
'BTCV_Cervix_ct',
'BraTS2021_mr_flair',
'BraTS2021_mr_t1',
'BraTS2021_mr_t1ce',
'BraTS2021_mr_t2',
'BrainPTM2021_mr_t1',
'BrainTumour_mr_flair',
'BrainTumour_mr_t1gd',
'BrainTumour_mr_t1w',
'BrainTumour_mr_t2w',
'CAUSE07_mr_unknown',
'COVID1920_ct',
'COVID19CTscans_ct',
'CTORG_ct',
'CTPelvic1k_ct',
'CTSpine1K_ct',
'Chest_CT_Scans_with_COVID-19_ct',
'FLARE21_ct',
'FeTA2022_mr_t2w',
'HeadandNeckAutoSegmentationChallenge_ct',
'HeartSegMRI_mr_unknown',
'ISLES2022_mr_adc',
'ISLES2022_mr_dwi',
'KiPA22_ct',
'LAScarQS22Task1_mr_lge',
'LAScarQS22Task2_mr_lge',
'LITS_ct',
'LNDb_ct',
'LUNA16_ct',
'LongitudinalMultipleSclerosisLesionSegmentation_mr_flair',
'LongitudinalMultipleSclerosisLesionSegmentation_mr_mprage',
'LongitudinalMultipleSclerosisLesionSegmentation_mr_pd',
'LongitudinalMultipleSclerosisLesionSegmentation_mr_t2',
'MESSEG_mr_flair',
'MMWHS_ct',
'MRBrain18_mr_t1',
'MRBrain18_mr_t1ir',
'MRBrain18_mr_t2flair',
'MRBrainS13_mr_t1',
'MRBrainS13_mr_t1ir',
'MRBrainS13_mr_t2flair',
'MSD02_Heart_mr_unknown',
'MSD04_Hippocampus_mr_unknown',
'MSD05_Prostate_mr_adc',
'MSD05_Prostate_mr_t2',
'MSD06_Lung_ct',
'MSD07_Pancreas_ct',
'MSD08_HepaticVessel_ct',
'MSD09_Spleen_ct',
'MSD10_Colon_ct',
'MSseg08_mr_flair',
'MSseg08_mr_t1',
'MSseg08_mr_t2',
'PROMISE12_mr_unknown',
'Prostate_MRI_Segmentation_Dataset_mr_t2w',
'RibFrac2020_ct',
'SLIVER07_ct',
'SegThor_ct',
'StructSeg2019_subtask1_ct',
'StructSeg2019_subtask2_ct',
'StructSeg2019_subtask3_ct',
'StructSeg2019_subtask4_ct',
'Totalsegmentator_dataset_ct',
'VALDO_Task2_mr_t2s',
'VESSEL2012_ct',
'WMH_mr_flair',
'WMH_mr_t1',
'WORD_ct',
'cSeg-2022_mr_unknown',
'iSeg2017_mr_t1',
'iSeg2017_mr_t2',
'iseg2019_mr_t1',
'iseg2019_mr_t2',
'mnms_mr_unknown',
]