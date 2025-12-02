# day2_loops.py
# Loops examples in Python

# For loop example
for i in range(1, 6):
    print(f"Number {i} squared is {i**2}")

# While loop example
num = 5
while num > 0:
    print(f"Countdown: {num}")
    num -= 1
print("Go!")

# Sum of numbers
total = 0
for i in range(1, 6):
    total += i
print("Sum from 1 to 5 =", total)
