class BiomarkerError(Exception):
    pass


organ_biomarker_quantifier_description = {
    "Name": "Universal Organ Biomarker Quantifier",
    "Property": "A quantification model",
    "Ability": "Measure the organ biomarker of the Image.",
    "Compulsory Input": "[Image] & [OrganObject] & [Organ Mask] & [Organ Label]",
    "Optional Input": "[OrganDim]",
    "Output": "[OrganDim] & [OrganQuant]" 
}

anomaly_biomarker_quantifier_description = {
    "Name": "Universal Anomaly Biomarker Quantifier",
    "Property": "A quantification model",
    "Ability": "Measure the anomaly biomarker of the Image.",
    "Compulsory Input": "[Image] & [AnomalyObject] & [Anomaly Mask] & [Anomaly Label]",
    "Optional Input": "[AnomalyDim]",
    "Output": "[AnomalyDim] & [AnomalyQuant]" 
}

def organ_biomarker_quantifier(input, reserved_dict):
    if '[Image]' in input and ('[Organ Mask]' in input and '[Organ Label]' in input):
        # reserved_dict['[Object]'] = 'PLACEHOLDER_OBJECT'
        # reserved_dict['[Dim]'] = 'PLACEHOLDER_DIM'
        reserved_dict['[OrganObject]'] = 'PLACEHOLDER_OBJECT'
        reserved_dict['[OrganDim]'] = 'PLACEHOLDER_DIM'
        reserved_dict['[OrganQuant]'] = 'PLACEHOLDER_QUANT'
    else:
        raise BiomarkerError("Universal Organ Biomarker Quantifier: Invalid input")
    
def anomaly_biomarker_quantifier(input, reserved_dict):
    if '[Image]' in input and ('[Anomaly Mask]' in input and '[Anomaly Label]' in input):
        # reserved_dict['[Object]'] = 'PLACEHOLDER_OBJECT'
        # reserved_dict['[Dim]'] = 'PLACEHOLDER_DIM'
        reserved_dict['[AnomalyObject]'] = 'PLACEHOLDER_OBJECT'
        reserved_dict['[AnomalyDim]'] = 'PLACEHOLDER_DIM'
        reserved_dict['[AnomalyQuant]'] = 'PLACEHOLDER_QUANT'
    else:
        raise BiomarkerError("Universal Anomaly Biomarker Quantifier: Invalid input")

    
BiomarkerQuantificationToolsDescription = {
    "Universal Organ Biomarker Quantifier": organ_biomarker_quantifier_description,
    "Universal Anomaly Biomarker Quantifier": anomaly_biomarker_quantifier_description
}

BiomarkerQuantificationTools = {
    "Universal Organ Biomarker Quantifier": organ_biomarker_quantifier,
    "Universal Anomaly Biomarker Quantifier": anomaly_biomarker_quantifier
}

