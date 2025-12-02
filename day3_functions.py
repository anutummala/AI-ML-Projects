# day3_functions.py
# Python Functions Examples

# 1. Simple function to greet a user
def greet_user(name):
    print(f"Hello, {name}!")

# Call the function
name = input("Enter your name: ")
greet_user(name)

# --------------------------------------------------

# 2. Function to check if a number is even
def is_even(num):
    if num % 2 == 0:
        return True
    else:
        return False

num = int(input("Enter a number: "))
if is_even(num):
    print(f"{num} is even")
else:
    print(f"{num} is odd")

# --------------------------------------------------

# 3. Function to sum numbers up to n
def sum_upto_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total

num = int(input("Enter a number to sum up to: "))
print(f"Sum from 1 to {num} =", sum_upto_n(num))

# --------------------------------------------------

# 4. Function to check password strength
def check_password_strength(password):
    length = len(password)
    if length < 6:
        return "Too short"
    elif 6 <= length <= 10:
        return "Medium strength"
    else:
        return "Strong password"

password = input("Enter a password: ")
print("Password strength:", check_password_strength(password))
