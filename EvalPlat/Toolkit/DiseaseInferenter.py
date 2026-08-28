class InferenceError(Exception):
    pass

disease_inferenter_description = {
    "Name": "Universal Disease Inferenter",
    "Property": "A Inference model",
    "Ability": "Infer disease based on organ segmentation and anomaly detection results.",
    "Compulsory Input": "[Image] & [Organ Mask] & [Organ Label] & [Anomaly Mask] & [Anomaly Label]",
    "Optional Input": "None",
    "Output": "[Disease]"
}

def disease_inferenter(input, reserved_dict):
    """
    Disease Inferenter
    A classification model
    Determine the disease of the Image.
    Input: [Image]
    Output: [Disease]
    """
    if '[Image]' in input and '[Organ Mask]' in input and '[Organ Label]' in input and '[Anomaly Mask]' in input and '[Anomaly Label]' in input:
        reserved_dict['[Disease]'] = 'PLACEHOLDER_DISEASE'
    else:
        raise InferenceError("Universal Disease Inferenter: Invalid input")

DiseaseInferenceToolsDescription = {
    "Universal Disease Inferenter": disease_inferenter_description
}

DiseaseInferenceTools = {
    "Universal Disease Inferenter": disease_inferenter
}