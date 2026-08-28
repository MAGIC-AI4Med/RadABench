class IndicatorError(Exception):
    pass

Indicator_evaluator_description = {
    "Name": "Universal Indicator Evaluator",
    "Property": "A evaluation model",
    "Ability": "Evaluate the indicator of the Image.",
    "Compulsory Input": "[Image] & [Indicator] & [Object] & [Dim] & [Quant]",
    "Optional Input": "[Object] & [Dim] & [Quant]",
    "Output": "[Value]"
}

def indicator_evaluator(input, reserved_dict):
    if '[Image]' in input and '[Indicator]' in input and '[Object]' in input and '[Dim]' in input and '[Quant]' in input:
        reserved_dict['[Value]'] = 'PLACEHOLDER_VALUE'
    else:
        raise IndicatorError("Universal Indicator Evaluator: Invalid input")
    
IndicatorEvaluationToolsDescription = {
    "Universal Indicator Evaluator": Indicator_evaluator_description
}

IndicatorEvaluationTools = {
    "Universal Indicator Evaluator": indicator_evaluator
}
