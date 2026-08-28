class AnatomyError(Exception):
    pass


anatomy_classifier_description = {
    "Name": "Universal Anatomy Classifier",
    "Property": "A classification model",
    "Ability": "Determine the anatomy of the Image.",
    "Compulsory Input": "[Image]",
    "Optional Input": "None",
    "Output": "[Anatomy]" 
}

def anatomy_classifier(input, reserved_dict):
    if '[Image]' in input:
        reserved_dict['[Anatomy]'] = 'PLACEHOLDER_ANATOMY'
    else:
        raise AnatomyError("Anatomy Classifier: Invalid Input")

AnatomyClassificationToolsDescription = {
    "Universal Anatomy Classifier": anatomy_classifier_description
}

AnatomyClassificationTools = {
    "Universal Anatomy Classifier": anatomy_classifier
}

