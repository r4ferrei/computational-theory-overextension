import sys

def is_mcdonough():
    return ('--mcdonough' in sys.argv)

def is_square():
    return ('--square' in sys.argv)

def is_square_mcdonough():
    return ('--square_mcdonough' in sys.argv)
