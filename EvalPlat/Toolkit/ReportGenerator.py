class GenerationError(Exception):
    pass

universal_report_generator_basic_description = {
    "Name": "Universal Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the modality and anatomy, generate a basic radiology report from the input image.",
    "Compulsory Input": "[Image] & [Modality] & [Anatomy]",
    "Optional Input": "None",
    "Output": "[Report]"
}

universal_report_generator_with_text_description = {
    "Name": "Universal Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the modality, anatomy and other text information (such as diseases, biomarkers, etc.), generate a radiology report from the input image.",
    "Compulsory Input": "[Image] & [Modality] & [Anatomy]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

universal_report_generator_with_mask_description = {
    "Name": "Universal Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the modality, anatomy and organ/anomaly masks and labels, generate a radiology report from the input image.",
    "Compulsory Input": "[Image] & [Modality] & [Anatomy]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

universal_report_generator_with_text_and_mask_description = {
    "Name": "Universal Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the modality, anatomy and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report from the input image.",
    "Compulsory Input": "[Image] & [Modality] & [Anatomy]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

headneck_xray_report_generator_basic_description = {
    "Name": "Headneck X-ray Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the head and neck X-ray image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

headneck_xray_report_generator_with_text_description = {
    "Name": "Headneck X-ray Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the head and neck X-ray image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

headneck_xray_report_generator_with_mask_description = {
    "Name": "Headneck X-ray Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the head and neck X-ray image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

headneck_xray_report_generator_with_text_and_mask_description = {
    "Name": "Headneck X-ray Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the head and neck X-ray image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

headneck_ct_report_generator_basic_description = {
    "Name": "Headneck CT Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the head and neck CT image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

headneck_ct_report_generator_with_text_description = {
    "Name": "Headneck CT Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the head and neck CT image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

headneck_ct_report_generator_with_mask_description = {
    "Name": "Headneck CT Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the head and neck CT image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

headneck_ct_report_generator_with_text_and_mask_description = {
    "Name": "Headneck CT Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the head and neck CT image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

headneck_mri_report_generator_basic_description = {
    "Name": "Headneck MRI Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the head and neck MRI image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

headneck_mri_report_generator_with_text_description = {
    "Name": "Headneck MRI Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the head and neck MRI image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

headneck_mri_report_generator_with_mask_description = {
    "Name": "Headneck MRI Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the head and neck MRI image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

headneck_mri_report_generator_with_text_and_mask_description = {
    "Name": "Headneck MRI Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the head and neck MRI image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

headneck_ultrasound_report_generator_basic_description = {
    "Name": "Headneck Ultrasound Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the head and neck ultrasound image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

headneck_ultrasound_report_generator_with_text_description = {
    "Name": "Headneck Ultrasound Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the head and neck ultrasound image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

headneck_ultrasound_report_generator_with_mask_description = {
    "Name": "Headneck Ultrasound Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the head and neck ultrasound image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

headneck_ultrasound_report_generator_with_text_and_mask_description = {
    "Name": "Headneck Ultrasound Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the head and neck ultrasound image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

chest_xray_report_generator_basic_description = {
    "Name": "Chest X-ray Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the chest X-ray image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

chest_xray_report_generator_with_text_description = {
    "Name": "Chest X-ray Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the chest X-ray image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

chest_xray_report_generator_with_mask_description = {
    "Name": "Chest X-ray Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the chest X-ray image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

chest_xray_report_generator_with_text_and_mask_description = {
    "Name": "Chest X-ray Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the chest X-ray image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

chest_ct_report_generator_basic_description = {
    "Name": "Chest CT Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the chest CT image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

chest_ct_report_generator_with_text_description = {
    "Name": "Chest CT Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the chest CT image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

chest_ct_report_generator_with_mask_description = {
    "Name": "Chest CT Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the chest CT image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

chest_ct_report_generator_with_text_and_mask_description = {
    "Name": "Chest CT Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the chest CT image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

chest_mri_report_generator_basic_description = {
    "Name": "Chest MRI Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the chest MRI image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

chest_mri_report_generator_with_text_description = {
    "Name": "Chest MRI Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the chest MRI image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

chest_mri_report_generator_with_mask_description = {
    "Name": "Chest MRI Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the chest MRI image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

chest_mri_report_generator_with_text_and_mask_description = {
    "Name": "Chest MRI Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the chest MRI image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

chest_ultrasound_report_generator_basic_description = {
    "Name": "Chest Ultrasound Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the chest ultrasound image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

chest_ultrasound_report_generator_with_text_description = {
    "Name": "Chest Ultrasound Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the chest ultrasound image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

chest_ultrasound_report_generator_with_mask_description = {
    "Name": "Chest Ultrasound Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the chest ultrasound image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

chest_ultrasound_report_generator_with_text_and_mask_description = {
    "Name": "Chest Ultrasound Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the chest ultrasound image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

breast_mammography_report_generator_basic_description = {
    "Name": "Breast Mammography Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the breast mammography image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

breast_mammography_report_generator_with_text_description = {
    "Name": "Breast Mammography Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the breast mammography image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

breast_mammography_report_generator_with_mask_description = {
    "Name": "Breast Mammography Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the breast mammography image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

breast_mammography_report_generator_with_text_and_mask_description = {
    "Name": "Breast Mammography Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the breast mammography image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

breast_ultrasound_report_generator_basic_description = {
    "Name": "Breast Ultrasound Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the breast ultrasound image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

breast_ultrasound_report_generator_with_text_description = {
    "Name": "Breast Ultrasound Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the breast ultrasound image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

breast_ultrasound_report_generator_with_mask_description = {
    "Name": "Breast Ultrasound Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the breast ultrasound image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

breast_ultrasound_report_generator_with_text_and_mask_description = {
    "Name": "Breast Ultrasound Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the breast ultrasound image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

breast_mri_report_generator_basic_description = {
    "Name": "Breast MRI Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the breast MRI image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

breast_mri_report_generator_with_text_description = {
    "Name": "Breast MRI Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the breast MRI image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

breast_mri_report_generator_with_mask_description = {
    "Name": "Breast MRI Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the breast MRI image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

breast_mri_report_generator_with_text_and_mask_description = {
    "Name": "Breast MRI Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the breast MRI image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

abdomenpelvis_xray_report_generator_basic_description = {
    "Name": "Abdomenpelvis X-ray Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis X-ray image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

abdomenpelvis_xray_report_generator_with_text_description = {
    "Name": "Abdomenpelvis X-ray Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis X-ray image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

abdomenpelvis_xray_report_generator_with_mask_description = {
    "Name": "Abdomenpelvis X-ray Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis X-ray image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

abdomenpelvis_xray_report_generator_with_text_and_mask_description = {
    "Name": "Abdomenpelvis X-ray Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis X-ray image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

abdomenpelvis_ct_report_generator_basic_description = {
    "Name": "Abdomenpelvis CT Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis CT image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

abdomenpelvis_ct_report_generator_with_text_description = {
    "Name": "Abdomenpelvis CT Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis CT image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

abdomenpelvis_ct_report_generator_with_mask_description = {
    "Name": "Abdomenpelvis CT Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis CT image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

abdomenpelvis_ct_report_generator_with_text_and_mask_description = {
    "Name": "Abdomenpelvis CT Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis CT image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

abdomenpelvis_mri_report_generator_basic_description = {
    "Name": "Abdomenpelvis MRI Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis MRI image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

abdomenpelvis_mri_report_generator_with_text_description = {
    "Name": "Abdomenpelvis MRI Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis MRI image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

abdomenpelvis_mri_report_generator_with_mask_description = {
    "Name": "Abdomenpelvis MRI Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis MRI image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

abdomenpelvis_mri_report_generator_with_text_and_mask_description = {
    "Name": "Abdomenpelvis MRI Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis MRI image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

abdomenpelvis_ultrasound_report_generator_basic_description = {
    "Name": "Abdomenpelvis Ultrasound Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis ultrasound image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

abdomenpelvis_ultrasound_report_generator_with_text_description = {
    "Name": "Abdomenpelvis Ultrasound Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis ultrasound image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

abdomenpelvis_ultrasound_report_generator_with_mask_description = {
    "Name": "Abdomenpelvis Ultrasound Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis ultrasound image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

abdomenpelvis_ultrasound_report_generator_with_text_and_mask_description = {
    "Name": "Abdomenpelvis Ultrasound Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the abdomenpelvis ultrasound image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

limb_xray_report_generator_basic_description = {
    "Name": "Limb X-ray Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the limb X-ray image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

limb_xray_report_generator_with_text_description = {
    "Name": "Limb X-ray Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the limb X-ray image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

limb_xray_report_generator_with_mask_description = {
    "Name": "Limb X-ray Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the limb X-ray image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

limb_xray_report_generator_with_text_and_mask_description = {
    "Name": "Limb X-ray Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the limb X-ray image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

limb_ct_report_generator_basic_description = {
    "Name": "Limb CT Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the limb CT image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

limb_ct_report_generator_with_text_description = {
    "Name": "Limb CT Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the limb CT image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

limb_ct_report_generator_with_mask_description = {
    "Name": "Limb CT Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the limb CT image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

limb_ct_report_generator_with_text_and_mask_description = {
    "Name": "Limb CT Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the limb CT image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

limb_mri_report_generator_basic_description = {
    "Name": "Limb MRI Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the limb MRI image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

limb_mri_report_generator_with_text_description = {
    "Name": "Limb MRI Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the limb MRI image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

limb_mri_report_generator_with_mask_description = {
    "Name": "Limb MRI Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the limb MRI image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

limb_mri_report_generator_with_text_and_mask_description = {
    "Name": "Limb MRI Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the limb MRI image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

limb_ultrasound_report_generator_basic_description = {
    "Name": "Limb Ultrasound Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the limb ultrasound image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

limb_ultrasound_report_generator_with_text_description = {
    "Name": "Limb Ultrasound Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the limb ultrasound image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

limb_ultrasound_report_generator_with_mask_description = {
    "Name": "Limb Ultrasound Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the limb ultrasound image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

limb_ultrasound_report_generator_with_text_and_mask_description = {
    "Name": "Limb Ultrasound Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the limb ultrasound image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

spine_xray_report_generator_basic_description = {
    "Name": "Spine X-ray Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the spine X-ray image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

spine_xray_report_generator_with_text_description = {
    "Name": "Spine X-ray Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the spine X-ray image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

spine_xray_report_generator_with_mask_description = {
    "Name": "Spine X-ray Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the spine X-ray image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

spine_xray_report_generator_with_text_and_mask_description = {
    "Name": "Spine X-ray Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the spine X-ray image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

spine_ct_report_generator_basic_description = {
    "Name": "Spine CT Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the spine CT image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

spine_ct_report_generator_with_text_description = {
    "Name": "Spine CT Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the spine CT image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

spine_ct_report_generator_with_mask_description = {
    "Name": "Spine CT Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the spine CT image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

spine_ct_report_generator_with_text_and_mask_description = {
    "Name": "Spine CT Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the spine CT image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

spine_mri_report_generator_basic_description = {
    "Name": "Spine MRI Report Generator Basic",
    "Property": "A report generation model",
    "Ability": "Given the spine MRI image, generate a basic radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Report]"
}

spine_mri_report_generator_with_text_description = {
    "Name": "Spine MRI Report Generator with Text",
    "Property": "A report generation model",
    "Ability": "Given the spine MRI image and other text information (such as diseases, biomarkers, etc.), generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

spine_mri_report_generator_with_mask_description = {
    "Name": "Spine MRI Report Generator with Mask",
    "Property": "A report generation model",
    "Ability": "Given the spine MRI image and organ/anomaly masks and labels, generate a radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Output": "[Report]"
}

spine_mri_report_generator_with_text_and_mask_description = {
    "Name": "Spine MRI Report Generator with Text and Mask",
    "Property": "A report generation model",
    "Ability": "Given the spine MRI image and all possible additional information (such as diseases, biomarkers, organ/anomaly masks and labels, etc.), generate a comprehensive radiology report.",
    "Compulsory Input": "[Image]",
    "Optional Input": "[Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Report]"
}

def universal_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and '[Modality]' in input and '[Anatomy]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Universal Report Generator Basic: Invalid input")

def universal_report_generator_with_text(input, reserved_dict):    
    if '[Image]' in input and '[Modality]' in input and '[Anatomy]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Universal Report Generator with Text: Invalid input")

def universal_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and '[Modality]' in input and '[Anatomy]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):  
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Universal Report Generator with Mask: Invalid input")
        
def universal_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input and '[Modality]' in input and '[Anatomy]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'  
    else:
        raise GenerationError("Universal Report Generator with Text and Mask: Invalid input")

def headneck_xray_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck X-ray Report Generator Basic: Invalid input")
        
def headneck_xray_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck X-ray Report Generator with Text: Invalid input") 

def headneck_xray_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck X-ray Report Generator with Mask: Invalid input")
        
def headneck_xray_report_generator_with_text_and_mask(input, reserved_dict):  
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck X-ray Report Generator with Text and Mask: Invalid input")
        
def headneck_ct_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck CT Report Generator Basic: Invalid input")

def headneck_ct_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck CT Report Generator with Text: Invalid input")

def headneck_ct_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck CT Report Generator with Mask: Invalid input")

def headneck_ct_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck CT Report Generator with Text and Mask: Invalid input")

def headneck_mri_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck MRI Report Generator Basic: Invalid input")

def headneck_mri_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck MRI Report Generator with Text: Invalid input")

def headneck_mri_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck MRI Report Generator with Mask: Invalid input")

def headneck_mri_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck MRI Report Generator with Text and Mask: Invalid input")

def headneck_ultrasound_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck Ultrasound Report Generator Basic: Invalid input")

def headneck_ultrasound_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck Ultrasound Report Generator with Text: Invalid input")

def headneck_ultrasound_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck Ultrasound Report Generator with Mask: Invalid input")

def headneck_ultrasound_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Headneck Ultrasound Report Generator with Text and Mask: Invalid input")

def chest_xray_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest X-ray Report Generator Basic: Invalid input")

def chest_xray_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest X-ray Report Generator with Text: Invalid input")

def chest_xray_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest X-ray Report Generator with Mask: Invalid input")

def chest_xray_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest X-ray Report Generator with Text and Mask: Invalid input")

def chest_ct_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest CT Report Generator Basic: Invalid input")

def chest_ct_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest CT Report Generator with Text: Invalid input")

def chest_ct_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest CT Report Generator with Mask: Invalid input")

def chest_ct_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest CT Report Generator with Text and Mask: Invalid input")

def chest_mri_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest MRI Report Generator Basic: Invalid input")

def chest_mri_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest MRI Report Generator with Text: Invalid input")

def chest_mri_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest MRI Report Generator with Mask: Invalid input")

def chest_mri_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest MRI Report Generator with Text and Mask: Invalid input")

def chest_ultrasound_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest Ultrasound Report Generator Basic: Invalid input")

def chest_ultrasound_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest Ultrasound Report Generator with Text: Invalid input")

def chest_ultrasound_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest Ultrasound Report Generator with Mask: Invalid input")

def chest_ultrasound_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Chest Ultrasound Report Generator with Text and Mask: Invalid input")

def breast_mammography_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast Mammography Report Generator Basic: Invalid input")

def breast_mammography_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast Mammography Report Generator with Text: Invalid input")

def breast_mammography_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast Mammography Report Generator with Mask: Invalid input")

def breast_mammography_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast Mammography Report Generator with Text and Mask: Invalid input")

def breast_ultrasound_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast Ultrasound Report Generator Basic: Invalid input")

def breast_ultrasound_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast Ultrasound Report Generator with Text: Invalid input")

def breast_ultrasound_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast Ultrasound Report Generator with Mask: Invalid input")

def breast_ultrasound_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast Ultrasound Report Generator with Text and Mask: Invalid input")

def breast_mri_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast MRI Report Generator Basic: Invalid input")

def breast_mri_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast MRI Report Generator with Text: Invalid input")

def breast_mri_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast MRI Report Generator with Mask: Invalid input")

def breast_mri_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Breast MRI Report Generator with Text and Mask: Invalid input")

def abdomenpelvis_xray_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis X-ray Report Generator Basic: Invalid input")

def abdomenpelvis_xray_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis X-ray Report Generator with Text: Invalid input")

def abdomenpelvis_xray_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis X-ray Report Generator with Mask: Invalid input")

def abdomenpelvis_xray_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis X-ray Report Generator with Text and Mask: Invalid input")

def abdomenpelvis_ct_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis CT Report Generator Basic: Invalid input")

def abdomenpelvis_ct_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis CT Report Generator with Text: Invalid input")

def abdomenpelvis_ct_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis CT Report Generator with Mask: Invalid input")

def abdomenpelvis_ct_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis CT Report Generator with Text and Mask: Invalid input")

def abdomenpelvis_mri_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis MRI Report Generator Basic: Invalid input")

def abdomenpelvis_mri_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis MRI Report Generator with Text: Invalid input")

def abdomenpelvis_mri_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis MRI Report Generator with Mask: Invalid input")

def abdomenpelvis_mri_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis MRI Report Generator with Text and Mask: Invalid input")

def abdomenpelvis_ultrasound_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis Ultrasound Report Generator Basic: Invalid input")

def abdomenpelvis_ultrasound_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis Ultrasound Report Generator with Text: Invalid input")

def abdomenpelvis_ultrasound_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis Ultrasound Report Generator with Mask: Invalid input")

def abdomenpelvis_ultrasound_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Abdomenpelvis Ultrasound Report Generator with Text and Mask: Invalid input")

def limb_xray_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb X-ray Report Generator Basic: Invalid input")

def limb_xray_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb X-ray Report Generator with Text: Invalid input")

def limb_xray_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb X-ray Report Generator with Mask: Invalid input")

def limb_xray_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb X-ray Report Generator with Text and Mask: Invalid input")

def limb_ct_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb CT Report Generator Basic: Invalid input")

def limb_ct_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb CT Report Generator with Text: Invalid input")

def limb_ct_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb CT Report Generator with Mask: Invalid input")

def limb_ct_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb CT Report Generator with Text and Mask: Invalid input")

def limb_mri_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb MRI Report Generator Basic: Invalid input")

def limb_mri_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb MRI Report Generator with Text: Invalid input")

def limb_mri_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb MRI Report Generator with Mask: Invalid input")

def limb_mri_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb MRI Report Generator with Text and Mask: Invalid input")

def limb_ultrasound_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb Ultrasound Report Generator Basic: Invalid input")

def limb_ultrasound_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb Ultrasound Report Generator with Text: Invalid input")

def limb_ultrasound_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb Ultrasound Report Generator with Mask: Invalid input")

def limb_ultrasound_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Limb Ultrasound Report Generator with Text and Mask: Invalid input")

def spine_xray_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine X-ray Report Generator Basic: Invalid input")

def spine_xray_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine X-ray Report Generator with Text: Invalid input")

def spine_xray_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine X-ray Report Generator with Mask: Invalid input")

def spine_xray_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine X-ray Report Generator with Text and Mask: Invalid input")

def spine_ct_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine CT Report Generator Basic: Invalid input")

def spine_ct_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine CT Report Generator with Text: Invalid input")

def spine_ct_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine CT Report Generator with Mask: Invalid input")

def spine_ct_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine CT Report Generator with Text and Mask: Invalid input")

def spine_mri_report_generator_basic(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine MRI Report Generator Basic: Invalid input")

def spine_mri_report_generator_with_text(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Organ Mask]', '[Organ Label]', '[Anomaly Mask]', '[Anomaly Label]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine MRI Report Generator with Text: Invalid input")

def spine_mri_report_generator_with_mask(input, reserved_dict):
    if '[Image]' in input and all(key not in input for key in ['[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine MRI Report Generator with Mask: Invalid input")

def spine_mri_report_generator_with_text_and_mask(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Report]'] = 'PLACEHOLDER_REPORT'
    else:
        raise GenerationError("Spine MRI Report Generator with Text and Mask: Invalid input")

ReportGenerationToolsDescription = {
"Universal Report Generator Basic": universal_report_generator_basic_description,
"Universal Report Generator with Text": universal_report_generator_with_text_description,
"Universal Report Generator with Mask": universal_report_generator_with_mask_description,
"Universal Report Generator with Text and Mask": universal_report_generator_with_text_and_mask_description,
"Headneck X-ray Report Generator Basic": headneck_xray_report_generator_basic_description,
"Headneck X-ray Report Generator with Text": headneck_xray_report_generator_with_text_description,
"Headneck X-ray Report Generator with Mask": headneck_xray_report_generator_with_mask_description,
"Headneck X-ray Report Generator with Text and Mask": headneck_xray_report_generator_with_text_and_mask_description,
"Headneck CT Report Generator Basic": headneck_ct_report_generator_basic_description,
"Headneck CT Report Generator with Text": headneck_ct_report_generator_with_text_description,
"Headneck CT Report Generator with Mask": headneck_ct_report_generator_with_mask_description,
"Headneck CT Report Generator with Text and Mask": headneck_ct_report_generator_with_text_and_mask_description,
"Headneck MRI Report Generator Basic": headneck_mri_report_generator_basic_description,
"Headneck MRI Report Generator with Text": headneck_mri_report_generator_with_text_description,
"Headneck MRI Report Generator with Mask": headneck_mri_report_generator_with_mask_description,
"Headneck MRI Report Generator with Text and Mask": headneck_mri_report_generator_with_text_and_mask_description,
"Headneck Ultrasound Report Generator Basic": headneck_ultrasound_report_generator_basic_description,
"Headneck Ultrasound Report Generator with Text": headneck_ultrasound_report_generator_with_text_description,
"Headneck Ultrasound Report Generator with Mask": headneck_ultrasound_report_generator_with_mask_description,
"Headneck Ultrasound Report Generator with Text and Mask": headneck_ultrasound_report_generator_with_text_and_mask_description,
"Chest X-ray Report Generator Basic": chest_xray_report_generator_basic_description,
"Chest X-ray Report Generator with Text": chest_xray_report_generator_with_text_description,
"Chest X-ray Report Generator with Mask": chest_xray_report_generator_with_mask_description,
"Chest X-ray Report Generator with Text and Mask": chest_xray_report_generator_with_text_and_mask_description,
"Chest CT Report Generator Basic": chest_ct_report_generator_basic_description,
"Chest CT Report Generator with Text": chest_ct_report_generator_with_text_description,
"Chest CT Report Generator with Mask": chest_ct_report_generator_with_mask_description,
"Chest CT Report Generator with Text and Mask": chest_ct_report_generator_with_text_and_mask_description,
"Chest MRI Report Generator Basic": chest_mri_report_generator_basic_description,
"Chest MRI Report Generator with Text": chest_mri_report_generator_with_text_description,
"Chest MRI Report Generator with Mask": chest_mri_report_generator_with_mask_description,
"Chest MRI Report Generator with Text and Mask": chest_mri_report_generator_with_text_and_mask_description,
"Chest Ultrasound Report Generator Basic": chest_ultrasound_report_generator_basic_description,
"Chest Ultrasound Report Generator with Text": chest_ultrasound_report_generator_with_text_description,
"Chest Ultrasound Report Generator with Mask": chest_ultrasound_report_generator_with_mask_description,
"Chest Ultrasound Report Generator with Text and Mask": chest_ultrasound_report_generator_with_text_and_mask_description,
"Breast Mammography Report Generator Basic": breast_mammography_report_generator_basic_description,
"Breast Mammography Report Generator with Text": breast_mammography_report_generator_with_text_description,
"Breast Mammography Report Generator with Mask": breast_mammography_report_generator_with_mask_description,
"Breast Mammography Report Generator with Text and Mask": breast_mammography_report_generator_with_text_and_mask_description,
"Breast Ultrasound Report Generator Basic": breast_ultrasound_report_generator_basic_description,
"Breast Ultrasound Report Generator with Text": breast_ultrasound_report_generator_with_text_description,
"Breast Ultrasound Report Generator with Mask": breast_ultrasound_report_generator_with_mask_description,
"Breast Ultrasound Report Generator with Text and Mask": breast_ultrasound_report_generator_with_text_and_mask_description,
"Breast MRI Report Generator Basic": breast_mri_report_generator_basic_description,
"Breast MRI Report Generator with Text": breast_mri_report_generator_with_text_description,
"Breast MRI Report Generator with Mask": breast_mri_report_generator_with_mask_description,
"Breast MRI Report Generator with Text and Mask": breast_mri_report_generator_with_text_and_mask_description,
"Abdomenpelvis X-ray Report Generator Basic": abdomenpelvis_xray_report_generator_basic_description,
"Abdomenpelvis X-ray Report Generator with Text": abdomenpelvis_xray_report_generator_with_text_description,
"Abdomenpelvis X-ray Report Generator with Mask": abdomenpelvis_xray_report_generator_with_mask_description,
"Abdomenpelvis X-ray Report Generator with Text and Mask": abdomenpelvis_xray_report_generator_with_text_and_mask_description,
"Abdomenpelvis CT Report Generator Basic": abdomenpelvis_ct_report_generator_basic_description,
"Abdomenpelvis CT Report Generator with Text": abdomenpelvis_ct_report_generator_with_text_description,
"Abdomenpelvis CT Report Generator with Mask": abdomenpelvis_ct_report_generator_with_mask_description,
"Abdomenpelvis CT Report Generator with Text and Mask": abdomenpelvis_ct_report_generator_with_text_and_mask_description,
"Abdomenpelvis MRI Report Generator Basic": abdomenpelvis_mri_report_generator_basic_description,
"Abdomenpelvis MRI Report Generator with Text": abdomenpelvis_mri_report_generator_with_text_description,
"Abdomenpelvis MRI Report Generator with Mask": abdomenpelvis_mri_report_generator_with_mask_description,
"Abdomenpelvis MRI Report Generator with Text and Mask": abdomenpelvis_mri_report_generator_with_text_and_mask_description,
"Abdomenpelvis Ultrasound Report Generator Basic": abdomenpelvis_ultrasound_report_generator_basic_description,
"Abdomenpelvis Ultrasound Report Generator with Text": abdomenpelvis_ultrasound_report_generator_with_text_description,
"Abdomenpelvis Ultrasound Report Generator with Mask": abdomenpelvis_ultrasound_report_generator_with_mask_description,
"Abdomenpelvis Ultrasound Report Generator with Text and Mask": abdomenpelvis_ultrasound_report_generator_with_text_and_mask_description,
"Limb X-ray Report Generator Basic": limb_xray_report_generator_basic_description,
"Limb X-ray Report Generator with Text": limb_xray_report_generator_with_text_description,
"Limb X-ray Report Generator with Mask": limb_xray_report_generator_with_mask_description,
"Limb X-ray Report Generator with Text and Mask": limb_xray_report_generator_with_text_and_mask_description,
"Limb CT Report Generator Basic": limb_ct_report_generator_basic_description,
"Limb CT Report Generator with Text": limb_ct_report_generator_with_text_description,
"Limb CT Report Generator with Mask": limb_ct_report_generator_with_mask_description,
"Limb CT Report Generator with Text and Mask": limb_ct_report_generator_with_text_and_mask_description,
"Limb MRI Report Generator Basic": limb_mri_report_generator_basic_description,
"Limb MRI Report Generator with Text": limb_mri_report_generator_with_text_description,
"Limb MRI Report Generator with Mask": limb_mri_report_generator_with_mask_description,
"Limb MRI Report Generator with Text and Mask": limb_mri_report_generator_with_text_and_mask_description,
"Limb Ultrasound Report Generator Basic": limb_ultrasound_report_generator_basic_description,
"Limb Ultrasound Report Generator with Text": limb_ultrasound_report_generator_with_text_description,
"Limb Ultrasound Report Generator with Mask": limb_ultrasound_report_generator_with_mask_description,
"Limb Ultrasound Report Generator with Text and Mask": limb_ultrasound_report_generator_with_text_and_mask_description,
"Spine X-ray Report Generator Basic": spine_xray_report_generator_basic_description,
"Spine X-ray Report Generator with Text": spine_xray_report_generator_with_text_description,
"Spine X-ray Report Generator with Mask": spine_xray_report_generator_with_mask_description,
"Spine X-ray Report Generator with Text and Mask": spine_xray_report_generator_with_text_and_mask_description,
"Spine CT Report Generator Basic": spine_ct_report_generator_basic_description,
"Spine CT Report Generator with Text": spine_ct_report_generator_with_text_description,
"Spine CT Report Generator with Mask": spine_ct_report_generator_with_mask_description,
"Spine CT Report Generator with Text and Mask": spine_ct_report_generator_with_text_and_mask_description,
"Spine MRI Report Generator Basic": spine_mri_report_generator_basic_description,
"Spine MRI Report Generator with Text": spine_mri_report_generator_with_text_description,
"Spine MRI Report Generator with Mask": spine_mri_report_generator_with_mask_description,
"Spine MRI Report Generator with Text and Mask": spine_mri_report_generator_with_text_and_mask_description,
}

ReportGenerationTools = {
    "Universal Report Generator Basic": universal_report_generator_basic,
    "Universal Report Generator with Text": universal_report_generator_with_text, 
    "Universal Report Generator with Mask": universal_report_generator_with_mask,
    "Universal Report Generator with Text and Mask": universal_report_generator_with_text_and_mask,
    "Headneck X-ray Report Generator Basic": headneck_xray_report_generator_basic,
    "Headneck X-ray Report Generator with Text": headneck_xray_report_generator_with_text,
    "Headneck X-ray Report Generator with Mask": headneck_xray_report_generator_with_mask, 
    "Headneck X-ray Report Generator with Text and Mask": headneck_xray_report_generator_with_text_and_mask,
    "Headneck CT Report Generator Basic": headneck_ct_report_generator_basic,
    "Headneck CT Report Generator with Text": headneck_ct_report_generator_with_text,
    "Headneck CT Report Generator with Mask": headneck_ct_report_generator_with_mask,
    "Headneck CT Report Generator with Text and Mask": headneck_ct_report_generator_with_text_and_mask,
    "Headneck MRI Report Generator Basic": headneck_mri_report_generator_basic,
    "Headneck MRI Report Generator with Text": headneck_mri_report_generator_with_text,
    "Headneck MRI Report Generator with Mask": headneck_mri_report_generator_with_mask,
    "Headneck MRI Report Generator with Text and Mask": headneck_mri_report_generator_with_text_and_mask,
    "Headneck Ultrasound Report Generator Basic": headneck_ultrasound_report_generator_basic,
    "Headneck Ultrasound Report Generator with Text": headneck_ultrasound_report_generator_with_text,
    "Headneck Ultrasound Report Generator with Mask": headneck_ultrasound_report_generator_with_mask,
    "Headneck Ultrasound Report Generator with Text and Mask": headneck_ultrasound_report_generator_with_text_and_mask,
    "Chest X-ray Report Generator Basic": chest_xray_report_generator_basic,
    "Chest X-ray Report Generator with Text": chest_xray_report_generator_with_text,
    "Chest X-ray Report Generator with Mask": chest_xray_report_generator_with_mask,
    "Chest X-ray Report Generator with Text and Mask": chest_xray_report_generator_with_text_and_mask,
    "Chest CT Report Generator Basic": chest_ct_report_generator_basic,
    "Chest CT Report Generator with Text": chest_ct_report_generator_with_text,
    "Chest CT Report Generator with Mask": chest_ct_report_generator_with_mask,
    "Chest CT Report Generator with Text and Mask": chest_ct_report_generator_with_text_and_mask,
    "Chest MRI Report Generator Basic": chest_mri_report_generator_basic,
    "Chest MRI Report Generator with Text": chest_mri_report_generator_with_text,
    "Chest MRI Report Generator with Mask": chest_mri_report_generator_with_mask,
    "Chest MRI Report Generator with Text and Mask": chest_mri_report_generator_with_text_and_mask,
    "Chest Ultrasound Report Generator Basic": chest_ultrasound_report_generator_basic,
    "Chest Ultrasound Report Generator with Text": chest_ultrasound_report_generator_with_text,
    "Chest Ultrasound Report Generator with Mask": chest_ultrasound_report_generator_with_mask,
    "Chest Ultrasound Report Generator with Text and Mask": chest_ultrasound_report_generator_with_text_and_mask,
    "Breast Mammography Report Generator Basic": breast_mammography_report_generator_basic,
    "Breast Mammography Report Generator with Text": breast_mammography_report_generator_with_text,
    "Breast Mammography Report Generator with Mask": breast_mammography_report_generator_with_mask,
    "Breast Mammography Report Generator with Text and Mask": breast_mammography_report_generator_with_text_and_mask,
    "Breast Ultrasound Report Generator Basic": breast_ultrasound_report_generator_basic,
    "Breast Ultrasound Report Generator with Text": breast_ultrasound_report_generator_with_text,
    "Breast Ultrasound Report Generator with Mask": breast_ultrasound_report_generator_with_mask,
    "Breast Ultrasound Report Generator with Text and Mask": breast_ultrasound_report_generator_with_text_and_mask,
    "Breast MRI Report Generator Basic": breast_mri_report_generator_basic,
    "Breast MRI Report Generator with Text": breast_mri_report_generator_with_text,
    "Breast MRI Report Generator with Mask": breast_mri_report_generator_with_mask,
    "Breast MRI Report Generator with Text and Mask": breast_mri_report_generator_with_text_and_mask,
    "Abdomenpelvis X-ray Report Generator Basic": abdomenpelvis_xray_report_generator_basic,
    "Abdomenpelvis X-ray Report Generator with Text": abdomenpelvis_xray_report_generator_with_text,
    "Abdomenpelvis X-ray Report Generator with Mask": abdomenpelvis_xray_report_generator_with_mask,
    "Abdomenpelvis X-ray Report Generator with Text and Mask": abdomenpelvis_xray_report_generator_with_text_and_mask,
    "Abdomenpelvis CT Report Generator Basic": abdomenpelvis_ct_report_generator_basic,
    "Abdomenpelvis CT Report Generator with Text": abdomenpelvis_ct_report_generator_with_text,
    "Abdomenpelvis CT Report Generator with Mask": abdomenpelvis_ct_report_generator_with_mask,
    "Abdomenpelvis CT Report Generator with Text and Mask": abdomenpelvis_ct_report_generator_with_text_and_mask,
    "Abdomenpelvis MRI Report Generator Basic": abdomenpelvis_mri_report_generator_basic,
    "Abdomenpelvis MRI Report Generator with Text": abdomenpelvis_mri_report_generator_with_text,
    "Abdomenpelvis MRI Report Generator with Mask": abdomenpelvis_mri_report_generator_with_mask,
    "Abdomenpelvis MRI Report Generator with Text and Mask": abdomenpelvis_mri_report_generator_with_text_and_mask,
    "Abdomenpelvis Ultrasound Report Generator Basic": abdomenpelvis_ultrasound_report_generator_basic,
    "Abdomenpelvis Ultrasound Report Generator with Text": abdomenpelvis_ultrasound_report_generator_with_text,
    "Abdomenpelvis Ultrasound Report Generator with Mask": abdomenpelvis_ultrasound_report_generator_with_mask,
    "Abdomenpelvis Ultrasound Report Generator with Text and Mask": abdomenpelvis_ultrasound_report_generator_with_text_and_mask,
    "Limb X-ray Report Generator Basic": limb_xray_report_generator_basic,
    "Limb X-ray Report Generator with Text": limb_xray_report_generator_with_text,
    "Limb X-ray Report Generator with Mask": limb_xray_report_generator_with_mask,
    "Limb X-ray Report Generator with Text and Mask": limb_xray_report_generator_with_text_and_mask,
    "Limb CT Report Generator Basic": limb_ct_report_generator_basic,
    "Limb CT Report Generator with Text": limb_ct_report_generator_with_text,
    "Limb CT Report Generator with Mask": limb_ct_report_generator_with_mask,
    "Limb CT Report Generator with Text and Mask": limb_ct_report_generator_with_text_and_mask,
    "Limb MRI Report Generator Basic": limb_mri_report_generator_basic,
    "Limb MRI Report Generator with Text": limb_mri_report_generator_with_text,
    "Limb MRI Report Generator with Mask": limb_mri_report_generator_with_mask,
    "Limb MRI Report Generator with Text and Mask": limb_mri_report_generator_with_text_and_mask,
    "Limb Ultrasound Report Generator Basic": limb_ultrasound_report_generator_basic,
    "Limb Ultrasound Report Generator with Text": limb_ultrasound_report_generator_with_text,
    "Limb Ultrasound Report Generator with Mask": limb_ultrasound_report_generator_with_mask,
    "Limb Ultrasound Report Generator with Text and Mask": limb_ultrasound_report_generator_with_text_and_mask,
    "Spine X-ray Report Generator Basic": spine_xray_report_generator_basic,
    "Spine X-ray Report Generator with Text": spine_xray_report_generator_with_text,
    "Spine X-ray Report Generator with Mask": spine_xray_report_generator_with_mask,
    "Spine X-ray Report Generator with Text and Mask": spine_xray_report_generator_with_text_and_mask,
    "Spine CT Report Generator Basic": spine_ct_report_generator_basic,
    "Spine CT Report Generator with Text": spine_ct_report_generator_with_text,
    "Spine CT Report Generator with Mask": spine_ct_report_generator_with_mask,
    "Spine CT Report Generator with Text and Mask": spine_ct_report_generator_with_text_and_mask,
    "Spine MRI Report Generator Basic": spine_mri_report_generator_basic,
    "Spine MRI Report Generator with Text": spine_mri_report_generator_with_text,
    "Spine MRI Report Generator with Mask": spine_mri_report_generator_with_mask,
    "Spine MRI Report Generator with Text and Mask": spine_mri_report_generator_with_text_and_mask,
}
