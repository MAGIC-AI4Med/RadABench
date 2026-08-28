class TreatmentError(Exception):
    pass

treatment_recommendator_description = {
    "Name": "Universal Treatment Recommendator",
    "Property": "A language model",
    "Ability": "Recommend the treatment of the Image.",
    "Compulsory Input": "[Information] & [Report]",
    "Optional Input": "[Anomaly Label] & [Disease] & [Object] & [Dim] & [Quant] & [Indicator] & [Value]",
    "Output": "[Value]"
}

def treatment_recommendator(input, reserved_dict):
    if '[Information]' in input and '[Report]' in input and all(key not in input for key in ['[Anomaly Label]', '[Disease]', '[Object]', '[Dim]', '[Quant]', '[Indicator]', '[Value]']):
        reserved_dict['[Value]'] = 'PLACEHOLDER_VALUE'
    else:
        raise TreatmentError("Universal Treatment Recommendator: Invalid input")
    
TreatmentRecommendationToolsDescription = { 
    "Universal Treatment Recommendator": treatment_recommendator_description
}

TreatmentRecommendationTools = {
    "Universal Treatment Recommendator": treatment_recommendator
}