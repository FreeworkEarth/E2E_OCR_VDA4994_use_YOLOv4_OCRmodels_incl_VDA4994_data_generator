import random
import string
from functools import*

""" random text/alphanumeric in ascii generator"""

def random_text_gen(length=32, randomascii=True, uppercase=True, lowercase=True, numbers=True):
    character_set = ''  # lowercase, uppercase, digits etc. possible
    if randomascii:
        character_set += string.ascii_letters
    elif uppercase:
        character_set += string.ascii_uppercase
    elif lowercase:
        character_set += string.ascii_lowercase
    elif numbers:
        character_set += string.digits

    return ''.join(random.choice(character_set) for i in range(length))