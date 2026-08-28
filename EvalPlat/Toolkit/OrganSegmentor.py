class SegmentationError(Exception):
    pass

universal_organ_segmentor_description = {
    "Name": "Universal Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the modality and anatomy, segment all organs in the Image.",
    "Compulsory Input": "[Image] & [Modality] & [Anatomy]", 
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

headneck_xray_organ_segmentor_description = {
    "Name": "Head and Neck X-ray Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Head and Neck X-ray Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None", 
    "Output": "[Organ Mask] & [Organ Label]"
}

headneck_ct_organ_segmentor_description = {
    "Name": "Head and Neck CT Organ Segmentor",
    "Property": "A segmentation model", 
    "Ability": "Given the Head and Neck CT Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

headneck_mri_organ_segmentor_description = {
    "Name": "Head and Neck MRI Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Head and Neck MRI Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

headneck_ultrasound_organ_segmentor_description = {
    "Name": "Head and Neck Ultrasound Organ Segmentor", 
    "Property": "A segmentation model",
    "Ability": "Given the Head and Neck Ultrasound Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

chest_xray_organ_segmentor_description = {
    "Name": "Chest X-ray Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Chest X-ray Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

chest_ct_organ_segmentor_description = {
    "Name": "Chest CT Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Chest CT Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

chest_mri_organ_segmentor_description = {
    "Name": "Chest MRI Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Chest MRI Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

chest_ultrasound_organ_segmentor_description = {
    "Name": "Chest Ultrasound Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Chest Ultrasound Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

breast_mammography_organ_segmentor_description = {
    "Name": "Breast Mammography Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Breast Mammography Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

breast_ultrasound_organ_segmentor_description = {
    "Name": "Breast Ultrasound Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Breast Ultrasound Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

breast_mri_organ_segmentor_description = {
    "Name": "Breast MRI Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Breast MRI Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

abdomenpelvis_xray_organ_segmentor_description = {
    "Name": "Abdomen and Pelvis X-ray Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Abdomen and Pelvis X-ray Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

abdomenpelvis_ct_organ_segmentor_description = {
    "Name": "Abdomen and Pelvis CT Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Abdomen and Pelvis CT Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

abdomenpelvis_mri_organ_segmentor_description = {
    "Name": "Abdomen and Pelvis MRI Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Abdomen and Pelvis MRI Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

abdomenpelvis_ultrasound_organ_segmentor_description = {
    "Name": "Abdomen and Pelvis Ultrasound Organ Segmentor",
    "Property": "A segmentation model", 
    "Ability": "Given the Abdomen and Pelvis Ultrasound Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

limb_xray_organ_segmentor_description = {
    "Name": "Limb X-ray Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Limb X-ray Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

limb_ct_organ_segmentor_description = {
    "Name": "Limb CT Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Limb CT Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

limb_mri_organ_segmentor_description = {
    "Name": "Limb MRI Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Limb MRI Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

limb_ultrasound_organ_segmentor_description = {
    "Name": "Limb Ultrasound Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Limb Ultrasound Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

spine_xray_organ_segmentor_description = {
    "Name": "Spine X-ray Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Spine X-ray Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

spine_ct_organ_segmentor_description = {
    "Name": "Spine CT Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Spine CT Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

spine_mri_organ_segmentor_description = {
    "Name": "Spine MRI Organ Segmentor",
    "Property": "A segmentation model",
    "Ability": "Given the Spine MRI Image, segment organs in it.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Organ Mask] & [Organ Label]"
}

def universal_organ_segmentor(input, reserved_dict):
    """
    Universal Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Image.
    """
    if '[Image]' in input and '[Modality]' in input and '[Anatomy]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Universal Organ Segmentor: Invalid input")

def headneck_xray_organ_segmentor(input, reserved_dict):
    """
    Head and Neck X-ray Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Head and Neck X-ray Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Head and Neck X-ray Organ Segmentor: Invalid input")
    
def headneck_ct_organ_segmentor(input, reserved_dict):
    """
    Head and Neck CT Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Head and Neck CT Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Head and Neck CT Organ Segmentor: Invalid input")
    
def headneck_mri_organ_segmentor(input, reserved_dict):
    """
    Head and Neck MRI Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Head and Neck MRI Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Head and Neck MRI Organ Segmentor: Invalid input")
    
def headneck_ultrasound_organ_segmentor(input, reserved_dict):
    """
    Head and Neck Ultrasound Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Head and Neck Ultrasound Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Head and Neck Ultrasound Organ Segmentor: Invalid input")
    
def chest_xray_organ_segmentor(input, reserved_dict):
    """
    Chest X-ray Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Chest X-ray Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Chest X-ray Organ Segmentor: Invalid input")
    
def chest_ct_organ_segmentor(input, reserved_dict):
    """
    Chest CT Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Chest CT Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Chest CT Organ Segmentor: Invalid input")
    
def chest_mri_organ_segmentor(input, reserved_dict):
    """
    Chest MRI Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Chest MRI Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Chest MRI Organ Segmentor: Invalid input")
    
def chest_ultrasound_organ_segmentor(input, reserved_dict):
    """
    Chest Ultrasound Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Chest Ultrasound Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Chest Ultrasound Organ Segmentor: Invalid input")
    
def breast_mammography_organ_segmentor(input, reserved_dict):
    """
    Breast Mammography Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Breast Mammography Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Breast Mammography Organ Segmentor: Invalid input")
    
def breast_ultrasound_organ_segmentor(input, reserved_dict):
    """
    Breast Ultrasound Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Breast Ultrasound Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Breast Ultrasound Organ Segmentor: Invalid input")
    
def breast_mri_organ_segmentor(input, reserved_dict):
    """
    Breast MRI Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Breast MRI Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Breast MRI Organ Segmentor: Invalid input")
    
def abdomenpelvis_xray_organ_segmentor(input, reserved_dict):
    """
    Abdomen and Pelvis X-ray Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Abdomen and Pelvis X-ray Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Abdomen and Pelvis X-ray Organ Segmentor: Invalid input")
    
def abdomenpelvis_ct_organ_segmentor(input, reserved_dict):
    """
    Abdomen and Pelvis CT Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Abdomen and Pelvis CT Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Abdomen and Pelvis CT Organ Segmentor: Invalid input")
    
def abdomenpelvis_mri_organ_segmentor(input, reserved_dict):
    """
    Abdomen and Pelvis MRI Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Abdomen and Pelvis MRI Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Abdomen and Pelvis MRI Organ Segmentor: Invalid input")
    
def abdomenpelvis_ultrasound_organ_segmentor(input, reserved_dict):
    """
    Abdomen and Pelvis Ultrasound Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Abdomen and Pelvis Ultrasound Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Abdomen and Pelvis Ultrasound Organ Segmentor: Invalid input")
    
def limb_xray_organ_segmentor(input, reserved_dict):
    """
    Limb X-ray Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Limb X-ray Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Limb X-ray Organ Segmentor: Invalid input")

def limb_ct_organ_segmentor(input, reserved_dict):
    """
    Limb CT Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Limb CT Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Limb CT Organ Segmentor: Invalid input")
    
def limb_mri_organ_segmentor(input, reserved_dict):
    """
    Limb MRI Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Limb MRI Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Limb MRI Organ Segmentor: Invalid input")
    
def limb_ultrasound_organ_segmentor(input, reserved_dict):
    """
    Limb Ultrasound Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Limb Ultrasound Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Limb Ultrasound Organ Segmentor: Invalid input")
    
def spine_xray_organ_segmentor(input, reserved_dict):
    """
    Spine X-ray Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Spine X-ray Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Spine X-ray Organ Segmentor: Invalid input")

def spine_ct_organ_segmentor(input, reserved_dict):
    """
    Spine CT Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Spine CT Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Spine CT Organ Segmentor: Invalid input")

def spine_mri_organ_segmentor(input, reserved_dict):
    """
    Spine MRI Organ Segmentor
    A segmentation model
    Given the modality and anatomy, segment organs in the Spine MRI Image.
    """
    if '[Image]' in input:
        reserved_dict['[Organ Mask]'] = 'PLACEHOLDER_ORGAN_MASK'
        reserved_dict['[Organ Label]'] = 'PLACEHOLDER_ORGAN_LABEL'
    else:
        raise SegmentationError("Spine MRI Organ Segmentor: Invalid input")

OrganSegmentationToolsDescription = {
    "Universal Organ Segmentor": universal_organ_segmentor_description,
    "Head and Neck X-ray Organ Segmentor": headneck_xray_organ_segmentor_description,
    "Head and Neck CT Organ Segmentor": headneck_ct_organ_segmentor_description,
    "Head and Neck MRI Organ Segmentor": headneck_mri_organ_segmentor_description,
    "Head and Neck Ultrasound Organ Segmentor": headneck_ultrasound_organ_segmentor_description,
    "Chest X-ray Organ Segmentor": chest_xray_organ_segmentor_description,
    "Chest CT Organ Segmentor": chest_ct_organ_segmentor_description,
    "Chest MRI Organ Segmentor": chest_mri_organ_segmentor_description,
    "Chest Ultrasound Organ Segmentor": chest_ultrasound_organ_segmentor_description,
    "Breast Mammography Organ Segmentor": breast_mammography_organ_segmentor_description,
    "Breast Ultrasound Organ Segmentor": breast_ultrasound_organ_segmentor_description,
    "Breast MRI Organ Segmentor": breast_mri_organ_segmentor_description,
    "Abdomen Pelvis X-ray Organ Segmentor": abdomenpelvis_xray_organ_segmentor_description,
    "Abdomen Pelvis CT Organ Segmentor": abdomenpelvis_ct_organ_segmentor_description,
    "Abdomen Pelvis MRI Organ Segmentor": abdomenpelvis_mri_organ_segmentor_description,
    "Abdomen Pelvis Ultrasound Organ Segmentor": abdomenpelvis_ultrasound_organ_segmentor_description,
    "Limb X-ray Organ Segmentor": limb_xray_organ_segmentor_description,
    "Limb CT Organ Segmentor": limb_ct_organ_segmentor_description,
    "Limb MRI Organ Segmentor": limb_mri_organ_segmentor_description,
    "Limb Ultrasound Organ Segmentor": limb_ultrasound_organ_segmentor_description,
    "Spine X-ray Organ Segmentor": spine_xray_organ_segmentor_description,
    "Spine CT Organ Segmentor": spine_ct_organ_segmentor_description,
    "Spine MRI Organ Segmentor": spine_mri_organ_segmentor_description,
}

OrganSegmentationTools = {
    "Universal Organ Segmentor": universal_organ_segmentor,
    "Head and Neck X-ray Organ Segmentor": headneck_xray_organ_segmentor,
    "Head and Neck CT Organ Segmentor": headneck_ct_organ_segmentor,
    "Head and Neck MRI Organ Segmentor": headneck_mri_organ_segmentor,
    "Head and Neck Ultrasound Organ Segmentor": headneck_ultrasound_organ_segmentor,
    "Chest X-ray Organ Segmentor": chest_xray_organ_segmentor,
    "Chest CT Organ Segmentor": chest_ct_organ_segmentor,
    "Chest MRI Organ Segmentor": chest_mri_organ_segmentor,
    "Chest Ultrasound Organ Segmentor": chest_ultrasound_organ_segmentor,
    "Breast Mammography Organ Segmentor": breast_mammography_organ_segmentor,
    "Breast Ultrasound Organ Segmentor": breast_ultrasound_organ_segmentor,
    "Breast MRI Organ Segmentor": breast_mri_organ_segmentor,
    "Abdomen Pelvis X-ray Organ Segmentor": abdomenpelvis_xray_organ_segmentor,
    "Abdomen Pelvis CT Organ Segmentor": abdomenpelvis_ct_organ_segmentor,
    "Abdomen Pelvis MRI Organ Segmentor": abdomenpelvis_mri_organ_segmentor,
    "Abdomen Pelvis Ultrasound Organ Segmentor": abdomenpelvis_ultrasound_organ_segmentor,
    "Limb X-ray Organ Segmentor": limb_xray_organ_segmentor,
    "Limb CT Organ Segmentor": limb_ct_organ_segmentor,
    "Limb MRI Organ Segmentor": limb_mri_organ_segmentor,
    "Limb Ultrasound Organ Segmentor": limb_ultrasound_organ_segmentor,
    "Spine X-ray Organ Segmentor": spine_xray_organ_segmentor,
    "Spine CT Organ Segmentor": spine_ct_organ_segmentor,
    "Spine MRI Organ Segmentor": spine_mri_organ_segmentor,
}


    
