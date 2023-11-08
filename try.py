import math
import numpy as np

# def sieve_of_eratosthenes(n):
#     sieve = [True] * (n + 1)
#     sieve[0:2] = [False, False]

#     for p in range(2, int(n ** 0.5) + 1):
#         if sieve[p]:
#             for i in range(p * p, n + 1, p):
#                 sieve[i] = False

#     primes = [i for i in range(n + 1) if sieve[i]]
#     return primes

# primes_up_to_100000 = sieve_of_eratosthenes(1000000)
# print(primes_up_to_100000)
# print(len(primes_up_to_100000))

def sieve_of_eratosthenes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    primes = []
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            primes.append(i)
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    for i in range(int(limit ** 0.5) + 1, limit + 1):
        if sieve[i]:
            primes.append(i)
    return primes

def next_good_number(number):
    limit = 2 * number  # Set the limit to reduce unnecessary calculations
    primes = sieve_of_eratosthenes(limit)
    print(len(primes))

    for p1 in primes:
        for p2 in primes:
            if p1 * p2 > number:
                return p1 * p2

# Input a number smaller than 10^10
input_number = int(input("Enter a number smaller than 10^10: "))
if input_number >= 10000000000:
    print("Please input a number smaller than 10^10.")
else:
    result = next_good_number(input_number)
    print(f"The next good number after {input_number} is {result}.")
