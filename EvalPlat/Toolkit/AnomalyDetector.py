class AnomalyError(Exception):
    pass

universal_anomaly_detector_description = {
    "Name": "Universal Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the modality and anatomy, determine the location and type of abnormality.",
    "Compulsory Input": "[Image] & [Modality] & [Anatomy]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}


headneck_xray_anomaly_detector_description = {
    "Name": "Headneck X-ray Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Head and Neck X-ray Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

headneck_ct_anomaly_detector_description = {
    "Name": "Headneck CT Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Head and Neck CT Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

headneck_mri_anomaly_detector_description = {
    "Name": "Headneck MRI Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Head and Neck MRI Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

headneck_ultrasound_anomaly_detector_description = {
    "Name": "Headneck Ultrasound Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Head and Neck Ultrasound Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

chest_xray_anomaly_detector_description = {
    "Name": "Chest X-ray Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Chest X-ray Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

chest_ct_anomaly_detector_description = {
    "Name": "Chest CT Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Chest CT Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

chest_mri_anomaly_detector_description = {
    "Name": "Chest MRI Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Chest MRI Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

chest_ultrasound_anomaly_detector_description = {
    "Name": "Chest Ultrasound Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Chest Ultrasound Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

breast_mammography_anomaly_detector_description = {
    "Name": "Breast Mammography Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Breast Mammography Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

breast_ultrasound_anomaly_detector_description = {
    "Name": "Breast Ultrasound Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Breast Ultrasound Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

breast_mri_anomaly_detector_description = {
    "Name": "Breast MRI Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Breast MRI Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

abdomenpelvis_xray_anomaly_detector_description = {
    "Name": "Abdomenpelvis X-ray Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Abdomenpelvis X-ray Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

abdomenpelvis_ct_anomaly_detector_description = {
    "Name": "Abdomenpelvis CT Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Abdomenpelvis CT Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

abdomenpelvis_mri_anomaly_detector_description = {
    "Name": "Abdomenpelvis MRI Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Abdomenpelvis MRI Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

abdomenpelvis_ultrasound_anomaly_detector_description = {
    "Name": "Abdomenpelvis Ultrasound Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Abdomenpelvis Ultrasound Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

limb_xray_anomaly_detector_description = {
    "Name": "Limb X-ray Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Limb X-ray Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

limb_ct_anomaly_detector_description = {
    "Name": "Limb CT Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Limb CT Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

limb_mri_anomaly_detector_description = {
    "Name": "Limb MRI Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Limb MRI Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

limb_ultrasound_anomaly_detector_description = {
    "Name": "Limb Ultrasound Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Limb Ultrasound Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

spine_xray_anomaly_detector_description = {
    "Name": "Spine X-ray Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Spine X-ray Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

spine_ct_anomaly_detector_description = {
    "Name": "Spine CT Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Spine CT Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

spine_mri_anomaly_detector_description = {
    "Name": "Spine MRI Anomaly Detector",
    "Property": "A detection model",
    "Ability": "Given the Spine MRI Image, determine the location and type of abnormality.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anomaly Mask] & [Anomaly Label]"
}

def universal_anomaly_detector(input, reserved_dict):
    """
    Anomaly Detection Model
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    Input: [Image] & [Modality] & [Anatomy]
    Output: [Anomaly Mask] & [Anomaly Label]
    """
    if '[Image]' in input and '[Modality]' in input and '[Anatomy]' in input:
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Universal Anomaly Detector: Invalid input")
    
def headneck_xray_anomaly_detector(input, reserved_dict):
    """
    Headneck X-ray Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Headneck X-ray Anomaly Detector: Invalid input")
    
def headneck_ct_anomaly_detector(input, reserved_dict):
    """
    Headneck CT Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Headneck CT Anomaly Detector: Invalid input")
    
def headneck_mri_anomaly_detector(input, reserved_dict):
    """
    Headneck MRI Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Headneck MRI Anomaly Detector: Invalid input")
    
def headneck_ultrasound_anomaly_detector(input, reserved_dict):
    """
    Headneck Ultrasound Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Headneck Ultrasound Anomaly Detector: Invalid input")
    
def chest_xray_anomaly_detector(input, reserved_dict):
    """
    Chest X-ray Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Chest X-ray Anomaly Detector: Invalid input")
    
def chest_ct_anomaly_detector(input, reserved_dict):
    """
    Chest CT Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Chest CT Anomaly Detector: Invalid input")
    
def chest_mri_anomaly_detector(input, reserved_dict):
    """
    Chest MRI Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Chest MRI Anomaly Detector: Invalid input")
    
def chest_ultrasound_anomaly_detector(input, reserved_dict):
    """
    Chest Ultrasound Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Chest Ultrasound Anomaly Detector: Invalid input")
    
def breast_mammography_anomaly_detector(input, reserved_dict):
    """
    Breast Mammography Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Breast Mammography Anomaly Detector: Invalid input")
    
def breast_ultrasound_anomaly_detector(input, reserved_dict):
    """
    Breast Ultrasound Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Breast Ultrasound Anomaly Detector: Invalid input")
    
def breast_mri_anomaly_detector(input, reserved_dict):
    """
    Breast MRI Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Breast MRI Anomaly Detector: Invalid input")
    
def abdomenpelvis_xray_anomaly_detector(input, reserved_dict):
    """
    Abdomenpelvis X-ray Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Abdomenpelvis X-ray Anomaly Detector: Invalid input")
    
def abdomenpelvis_ct_anomaly_detector(input, reserved_dict):
    """
    Abdomenpelvis CT Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Abdomenpelvis CT Anomaly Detector: Invalid input")
    
def abdomenpelvis_mri_anomaly_detector(input, reserved_dict):
    """
    Abdomenpelvis MRI Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Abdomenpelvis MRI Anomaly Detector: Invalid input")
    
def abdomenpelvis_ultrasound_anomaly_detector(input, reserved_dict):
    """
    Abdomenpelvis Ultrasound Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Abdomenpelvis Ultrasound Anomaly Detector: Invalid input")
    
def limb_xray_anomaly_detector(input, reserved_dict):
    """
    Limb X-ray Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Limb X-ray Anomaly Detector: Invalid input")
    
def limb_ct_anomaly_detector(input, reserved_dict):
    """
    Limb CT Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Limb CT Anomaly Detector: Invalid input")
    
def limb_mri_anomaly_detector(input, reserved_dict):
    """
    Limb MRI Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Limb MRI Anomaly Detector: Invalid input")
    
def limb_ultrasound_anomaly_detector(input, reserved_dict):
    """
    Limb Ultrasound Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Limb Ultrasound Anomaly Detector: Invalid input")
    
def spine_xray_anomaly_detector(input, reserved_dict):
    """
    Spine X-ray Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Spine X-ray Anomaly Detector: Invalid input")
    
def spine_ct_anomaly_detector(input, reserved_dict):
    """
    Spine CT Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Spine CT Anomaly Detector: Invalid input")
    
def spine_mri_anomaly_detector(input, reserved_dict):
    """
    Spine MRI Anomaly Detector
    A detection model
    Given the modality and anatomy, determine the location and type of abnormality.
    """
    if '[Image]' in input :
        reserved_dict['[Anomaly Mask]'] = 'PLACEHOLDER_ANOMALY_MASK'
        reserved_dict['[Anomaly Label]'] = 'PLACEHOLDER_ANOMALY_LABEL'
    else:
        raise AnomalyError("Spine MRI Anomaly Detector: Invalid input")
    
AnomalyDetectionToolsDescription = {
    "Universal Anomaly Detector": universal_anomaly_detector_description,
    "Headneck X-ray Anomaly Detector": headneck_xray_anomaly_detector_description,
    "Headneck CT Anomaly Detector": headneck_ct_anomaly_detector_description,
    "Headneck MRI Anomaly Detector": headneck_mri_anomaly_detector_description,
    "Headneck Ultrasound Anomaly Detector": headneck_ultrasound_anomaly_detector_description,
    "Chest X-ray Anomaly Detector": chest_xray_anomaly_detector_description,
    "Chest CT Anomaly Detector": chest_ct_anomaly_detector_description,
    "Chest MRI Anomaly Detector": chest_mri_anomaly_detector_description,
    "Chest Ultrasound Anomaly Detector": chest_ultrasound_anomaly_detector_description,
    "Breast Mammography Anomaly Detector": breast_mammography_anomaly_detector_description,
    "Breast Ultrasound Anomaly Detector": breast_ultrasound_anomaly_detector_description,
    "Breast MRI Anomaly Detector": breast_mri_anomaly_detector_description,
    "Abdomenpelvis X-ray Anomaly Detector": abdomenpelvis_xray_anomaly_detector_description,
    "Abdomenpelvis CT Anomaly Detector": abdomenpelvis_ct_anomaly_detector_description,
    "Abdomenpelvis MRI Anomaly Detector": abdomenpelvis_mri_anomaly_detector_description,
    "Abdomenpelvis Ultrasound Anomaly Detector": abdomenpelvis_ultrasound_anomaly_detector_description,
    "Limb X-ray Anomaly Detector": limb_xray_anomaly_detector_description,
    "Limb CT Anomaly Detector": limb_ct_anomaly_detector_description,
    "Limb MRI Anomaly Detector": limb_mri_anomaly_detector_description,
    "Limb Ultrasound Anomaly Detector": limb_ultrasound_anomaly_detector_description,
    "Spine X-ray Anomaly Detector": spine_xray_anomaly_detector_description,
    "Spine CT Anomaly Detector": spine_ct_anomaly_detector_description,
    "Spine MRI Anomaly Detector": spine_mri_anomaly_detector_description
}
    
AnomalyDetectionTools = {
    "Universal Anomaly Detector": universal_anomaly_detector,
    "Headneck X-ray Anomaly Detector": headneck_xray_anomaly_detector,
    "Headneck CT Anomaly Detector": headneck_ct_anomaly_detector,
    "Headneck MRI Anomaly Detector": headneck_mri_anomaly_detector,
    "Headneck Ultrasound Anomaly Detector": headneck_ultrasound_anomaly_detector,
    "Chest X-ray Anomaly Detector": chest_xray_anomaly_detector,
    "Chest CT Anomaly Detector": chest_ct_anomaly_detector,
    "Chest MRI Anomaly Detector": chest_mri_anomaly_detector,
    "Chest Ultrasound Anomaly Detector": chest_ultrasound_anomaly_detector,
    "Breast Mammography Anomaly Detector": breast_mammography_anomaly_detector,
    "Breast Ultrasound Anomaly Detector": breast_ultrasound_anomaly_detector,
    "Breast MRI Anomaly Detector": breast_mri_anomaly_detector,
    "Abdomenpelvis X-ray Anomaly Detector": abdomenpelvis_xray_anomaly_detector,
    "Abdomenpelvis CT Anomaly Detector": abdomenpelvis_ct_anomaly_detector,
    "Abdomenpelvis MRI Anomaly Detector": abdomenpelvis_mri_anomaly_detector,
    "Abdomenpelvis Ultrasound Anomaly Detector": abdomenpelvis_ultrasound_anomaly_detector,
    "Limb X-ray Anomaly Detector": limb_xray_anomaly_detector,
    "Limb CT Anomaly Detector": limb_ct_anomaly_detector,
    "Limb MRI Anomaly Detector": limb_mri_anomaly_detector,
    "Limb Ultrasound Anomaly Detector": limb_ultrasound_anomaly_detector,
    "Spine X-ray Anomaly Detector": spine_xray_anomaly_detector,
    "Spine CT Anomaly Detector": spine_ct_anomaly_detector,
    "Spine MRI Anomaly Detector": spine_mri_anomaly_detector
}

