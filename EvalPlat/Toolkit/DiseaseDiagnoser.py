class DiagnosisError(Exception):
    pass

universal_disease_diagnoser_description = {
    "Name": "Universal Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the modality and anatomy, diagnose diseases directly from the input image.",
    "Compulsory Input": "[Image] & [Modality] & [Anatomy]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

headneck_xray_disease_diagnoser_description = {
    "Name": "Headneck X-ray Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Head and Neck X-ray Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]",
}

headneck_ct_disease_diagnoser_description = {
    "Name": "Headneck CT Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Head and Neck CT Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

headneck_mri_disease_diagnoser_description = {
    "Name": "Headneck MRI Disease Diagnoser", 
    "Property": "A classification model",
    "Ability": "Given the Head and Neck MRI Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

headneck_ultrasound_disease_diagnoser_description = {
    "Name": "Headneck Ultrasound Disease Diagnoser",
    "Property": "A classification model", 
    "Ability": "Given the Head and Neck Ultrasound Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

chest_xray_disease_diagnoser_description = {
    "Name": "Chest X-ray Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Chest X-ray Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None", 
    "Output": "[Disease]"
}

chest_ct_disease_diagnoser_description = {
    "Name": "Chest CT Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Chest CT Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

chest_mri_disease_diagnoser_description = {
    "Name": "Chest MRI Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Chest MRI Image, diagnose diseases directly.", 
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

chest_ultrasound_disease_diagnoser_description = {
    "Name": "Chest Ultrasound Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Chest Ultrasound Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

breast_mammography_disease_diagnoser_description = {
    "Name": "Breast Mammography Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Breast Mammography Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

breast_ultrasound_disease_diagnoser_description = {
    "Name": "Breast Ultrasound Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Breast Ultrasound Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

breast_mri_disease_diagnoser_description = {
    "Name": "Breast MRI Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Breast MRI Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

abdomenpelvis_xray_disease_diagnoser_description = {
    "Name": "Abdomenpelvis X-ray Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Abdomenpelvis X-ray Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

abdomenpelvis_ct_disease_diagnoser_description = {
    "Name": "Abdomenpelvis CT Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Abdomenpelvis CT Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

abdomenpelvis_mri_disease_diagnoser_description = {
    "Name": "Abdomenpelvis MRI Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Abdomenpelvis MRI Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

abdomenpelvis_ultrasound_disease_diagnoser_description = {
    "Name": "Abdomenpelvis Ultrasound Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Abdomenpelvis Ultrasound Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

limb_xray_disease_diagnoser_description = {
    "Name": "Limb X-ray Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Limb X-ray Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

limb_ct_disease_diagnoser_description = {
    "Name": "Limb CT Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Limb CT Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

limb_mri_disease_diagnoser_description = {
    "Name": "Limb MRI Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Limb MRI Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

limb_ultrasound_disease_diagnoser_description = {
    "Name": "Limb Ultrasound Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Limb Ultrasound Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

spine_xray_disease_diagnoser_description = {
    "Name": "Spine X-ray Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Spine X-ray Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

spine_ct_disease_diagnoser_description = {
    "Name": "Spine CT Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Spine CT Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

spine_mri_disease_diagnoser_description = {
    "Name": "Spine MRI Disease Diagnoser",
    "Property": "A classification model",
    "Ability": "Given the Spine MRI Image, diagnose diseases directly.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

def universal_disease_diagnoser(input, reserved_dict):
    """
    Disease Diagnosis Model
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    Input: [Image] & [Modality] & [Anatomy]
    Output: [Diseases]
    """
    if '[Image]' in input and '[Modality]' in input and '[Anatomy]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Universal Disease Diagnoser: Invalid input")

def headneck_xray_disease_diagnoser(input, reserved_dict):
    """
    Headneck X-ray Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Headneck X-ray Disease Diagnoser: Invalid input")

def headneck_ct_disease_diagnoser(input, reserved_dict):
    """
    Headneck CT Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Headneck CT Disease Diagnoser: Invalid input")

def headneck_mri_disease_diagnoser(input, reserved_dict):
    """
    Headneck MRI Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Headneck MRI Disease Diagnoser: Invalid input")

def headneck_ultrasound_disease_diagnoser(input, reserved_dict):
    """
    Headneck Ultrasound Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Headneck Ultrasound Disease Diagnoser: Invalid input")

def chest_xray_disease_diagnoser(input, reserved_dict):
    """
    Chest X-ray Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Chest X-ray Disease Diagnoser: Invalid input")

def chest_ct_disease_diagnoser(input, reserved_dict):
    """
    Chest CT Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Chest CT Disease Diagnoser: Invalid input")

def chest_mri_disease_diagnoser(input, reserved_dict):
    """
    Chest MRI Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Chest MRI Disease Diagnoser: Invalid input")

def chest_ultrasound_disease_diagnoser(input, reserved_dict):
    """
    Chest Ultrasound Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Chest Ultrasound Disease Diagnoser: Invalid input")

def breast_mammography_disease_diagnoser(input, reserved_dict):
    """
    Breast Mammography Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Breast Mammography Disease Diagnoser: Invalid input")

def breast_ultrasound_disease_diagnoser(input, reserved_dict):
    """
    Breast Ultrasound Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Breast Ultrasound Disease Diagnoser: Invalid input")

def breast_mri_disease_diagnoser(input, reserved_dict):
    """
    Breast MRI Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Breast MRI Disease Diagnoser: Invalid input")

def abdomenpelvis_xray_disease_diagnoser(input, reserved_dict):
    """
    Abdomenpelvis X-ray Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Abdomenpelvis X-ray Disease Diagnoser: Invalid input")

def abdomenpelvis_ct_disease_diagnoser(input, reserved_dict):
    """
    Abdomenpelvis CT Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Abdomenpelvis CT Disease Diagnoser: Invalid input")

def abdomenpelvis_mri_disease_diagnoser(input, reserved_dict):
    """
    Abdomenpelvis MRI Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Abdomenpelvis MRI Disease Diagnoser: Invalid input")

def abdomenpelvis_ultrasound_disease_diagnoser(input, reserved_dict):
    """
    Abdomenpelvis Ultrasound Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Abdomenpelvis Ultrasound Disease Diagnoser: Invalid input")

def limb_xray_disease_diagnoser(input, reserved_dict):
    """
    Limb X-ray Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Limb X-ray Disease Diagnoser: Invalid input")

def limb_ct_disease_diagnoser(input, reserved_dict):
    """
    Limb CT Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Limb CT Disease Diagnoser: Invalid input")

def limb_mri_disease_diagnoser(input, reserved_dict):
    """
    Limb MRI Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Limb MRI Disease Diagnoser: Invalid input")

def limb_ultrasound_disease_diagnoser(input, reserved_dict):
    """
    Limb Ultrasound Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Limb Ultrasound Disease Diagnoser: Invalid input")

def spine_xray_disease_diagnoser(input, reserved_dict):
    """
    Spine X-ray Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Spine X-ray Disease Diagnoser: Invalid input")

def spine_ct_disease_diagnoser(input, reserved_dict):
    """
    Spine CT Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Spine CT Disease Diagnoser: Invalid input")

def spine_mri_disease_diagnoser(input, reserved_dict):
    """
    Spine MRI Disease Diagnoser
    A classification model
    Given the modality and anatomy, diagnose diseases directly from the input image.
    """
    if '[Image]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASES'
    else:
        raise DiagnosisError("Spine MRI Disease Diagnoser: Invalid input")
    



DiseaseDiagnosisTools = {
    "Universal Disease Diagnoser": universal_disease_diagnoser,
    "Headneck X-ray Disease Diagnoser": headneck_xray_disease_diagnoser,
    "Headneck CT Disease Diagnoser": headneck_ct_disease_diagnoser,
    "Headneck MRI Disease Diagnoser": headneck_mri_disease_diagnoser,
    "Headneck Ultrasound Disease Diagnoser": headneck_ultrasound_disease_diagnoser,
    "Chest X-ray Disease Diagnoser": chest_xray_disease_diagnoser,
    "Chest CT Disease Diagnoser": chest_ct_disease_diagnoser,
    "Chest MRI Disease Diagnoser": chest_mri_disease_diagnoser,
    "Chest Ultrasound Disease Diagnoser": chest_ultrasound_disease_diagnoser,
    "Breast Mammography Disease Diagnoser": breast_mammography_disease_diagnoser,
    "Breast Ultrasound Disease Diagnoser": breast_ultrasound_disease_diagnoser,
    "Breast MRI Disease Diagnoser": breast_mri_disease_diagnoser,
    "Abdomenpelvis X-ray Disease Diagnoser": abdomenpelvis_xray_disease_diagnoser,
    "Abdomenpelvis CT Disease Diagnoser": abdomenpelvis_ct_disease_diagnoser,
    "Abdomenpelvis MRI Disease Diagnoser": abdomenpelvis_mri_disease_diagnoser,
    "Abdomenpelvis Ultrasound Disease Diagnoser": abdomenpelvis_ultrasound_disease_diagnoser,
    "Limb X-ray Disease Diagnoser": limb_xray_disease_diagnoser,
    "Limb CT Disease Diagnoser": limb_ct_disease_diagnoser,
    "Limb MRI Disease Diagnoser": limb_mri_disease_diagnoser,
    "Limb Ultrasound Disease Diagnoser": limb_ultrasound_disease_diagnoser,
    "Spine X-ray Disease Diagnoser": spine_xray_disease_diagnoser,
    "Spine CT Disease Diagnoser": spine_ct_disease_diagnoser,
    "Spine MRI Disease Diagnoser": spine_mri_disease_diagnoser
}

DiseaseDiagnosisToolsDescription = {
    "Universal Disease Diagnoser": universal_disease_diagnoser_description,
    "Headneck X-ray Disease Diagnoser": headneck_xray_disease_diagnoser_description,
    "Headneck CT Disease Diagnoser": headneck_ct_disease_diagnoser_description,
    "Headneck MRI Disease Diagnoser": headneck_mri_disease_diagnoser_description,
    "Headneck Ultrasound Disease Diagnoser": headneck_ultrasound_disease_diagnoser_description,
    "Chest X-ray Disease Diagnoser": chest_xray_disease_diagnoser_description,
    "Chest CT Disease Diagnoser": chest_ct_disease_diagnoser_description,
    "Chest MRI Disease Diagnoser": chest_mri_disease_diagnoser_description,
    "Chest Ultrasound Disease Diagnoser": chest_ultrasound_disease_diagnoser_description,
    "Breast Mammography Disease Diagnoser": breast_mammography_disease_diagnoser_description,
    "Breast Ultrasound Disease Diagnoser": breast_ultrasound_disease_diagnoser_description,
    "Breast MRI Disease Diagnoser": breast_mri_disease_diagnoser_description,
    "Abdomenpelvis X-ray Disease Diagnoser": abdomenpelvis_xray_disease_diagnoser_description,
    "Abdomenpelvis CT Disease Diagnoser": abdomenpelvis_ct_disease_diagnoser_description,
    "Abdomenpelvis MRI Disease Diagnoser": abdomenpelvis_mri_disease_diagnoser_description,
    "Abdomenpelvis Ultrasound Disease Diagnoser": abdomenpelvis_ultrasound_disease_diagnoser_description,
    "Limb X-ray Disease Diagnoser": limb_xray_disease_diagnoser_description,
    "Limb CT Disease Diagnoser": limb_ct_disease_diagnoser_description,
    "Limb MRI Disease Diagnoser": limb_mri_disease_diagnoser_description,
    "Limb Ultrasound Disease Diagnoser": limb_ultrasound_disease_diagnoser_description,
    "Spine X-ray Disease Diagnoser": spine_xray_disease_diagnoser_description,
    "Spine CT Disease Diagnoser": spine_ct_disease_diagnoser_description,
    "Spine MRI Disease Diagnoser": spine_mri_disease_diagnoser_description
}