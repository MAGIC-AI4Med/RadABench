class ModalityError(Exception):
    pass

modality_classifier_description = {
    "Name": "Universal Modality Classifier",
    "Property": "A classification model",
    "Ability": "Determine the modality of the Image.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Modality]" 
}

def modality_classifier(input, reserved_dict):
    """
    Modality Classifier
    A classification model
    Determine the modality of the Image.
    Input: [Image]
    Output: [Modality]
    """
    if '[Image]' in input:
        reserved_dict['[Modality]'] = 'PLACEHOLDER_MODALITY'
    else:
        raise OSError("Modality Classifier: Invalid Input")

ModalityClassificationToolsDescription = {
    "Universal Modality Classifier": modality_classifier_description
}

ModalityClassificationTools = {
    "Universal Modality Classifier": modality_classifier
}



