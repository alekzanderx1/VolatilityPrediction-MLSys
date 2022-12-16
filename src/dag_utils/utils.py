# Helper function to convert number strings(ex 1.5M) into float or int datatype
def value_to_float(x):
    if type(x) == float or type(x) == int:
        return x
    if 'M' in x:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    return 0.0
