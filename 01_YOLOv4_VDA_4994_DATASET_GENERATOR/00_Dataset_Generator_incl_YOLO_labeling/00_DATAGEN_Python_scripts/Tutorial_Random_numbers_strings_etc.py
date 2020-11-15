""" Tut to use random"""
import random
import string
from functools import*


""" text generator"""
def random_text_gen(length=32, randomascii=True, uppercase=True, lowercase=True, numbers=True):
    character_set = ''  # lowercase, uppercase, digits etc. possile
    if randomascii:
        character_set += string.ascii_letters
    elif uppercase:
        character_set += string.ascii_uppercase
    elif lowercase:
        character_set += string.ascii_lowercase
    elif numbers:
        character_set += string.digits

    return ''.join(random.choice(character_set) for i in range(length))

# """ Random Number """
print(random.uniform(100,200))                              # random number between 100 and 200
print(random.randrange(100,200,2))                          # random number between 100 and 200 stepsize 2
print(random.randrange(100,200,5))                          # random number between 100 and 200 stepsize 5


number_list     = [1,2,3,4,5]
number_tuple    = (1,2,3,4,5)
number_string   = 'abcdef'

# """ Random Choice"""
print(random.choice(number_list))
print(random.choice(number_tuple))
print(random.choice(number_string))



"""Random shuffle"""
random.shuffle(number_list)
shuffled = random.sample(number_list, len(number_list))
sample   = random.sample(number_list, 3)
print(number_list)
print(shuffled)
print(sample)

""" random string"""
def random_all(length=32, randomascii=True, uppercase=True, lowercase=True, numbers=True):

    character_set = ''                                                  # lowercase, uppercase, digits etc. possile
    if randomascii:
        character_set += string.ascii_letters
    elif uppercase:
        character_set += string.ascii_uppercase
    elif lowercase:
        character_set += string.ascii_lowercase
    elif numbers:
        character_set += string.digits

    return ''.join(random.choice(character_set) for i in range(length))


my_random_string = random_all()
my_random_string_ten = random_all(length=10)
my_random_string_upper = random_all(10, randomascii=False, uppercase=True)
my_random_number = random_all(10, randomascii=False, uppercase=False, lowercase=False)
                                   
print(my_random_string)
print(my_random_string_ten)
print(my_random_string_upper)
print(my_random_number)

""" calculate random numbers with write automaticallly in dictionary"""
SIMULATIONS = 1000
possible_values = list(range(2, 13))
outcomes = {i: 0 for i in possible_values}

for i in range(SIMULATIONS):
    d1 = random.randint(1,6)
    d2 = random.randint(1,6)

    outcome = d1 + d2
    outcomes[outcome] += 1

print(outcomes)

check_accuracy = reduce((lambda x, y: x + y), outcomes.values())
print(check_accuracy)


"""Usecase Intrafly"""
number_list_yellow = [1,2]
distance = random.randint(0, 3840-300)
print(distance)