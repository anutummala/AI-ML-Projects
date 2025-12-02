# day4_lists_dicts.py
# Python Lists and Dictionaries Examples

# --------------------------
# 1. Working with Lists
numbers = [1, 2, 3, 4, 5, 6]
print("Original list:", numbers)

# Square each number
for n in numbers:
    print(f"Square of {n} is {n**2}")

# Sum of numbers
def sum_list(nums):
    total = 0
    for n in nums:
        total += n
    return total

print("Sum of [3, 5, 7]:", sum_list([3, 5, 7]))

# Add and remove elements
fruits = ["apple", "peach", "grapes"]
fruits.append("plum")
fruits.append("mango")
fruits.remove("plum")
print("Fruits list:", fruits)
print("Number of fruits:", len(fruits))

# --------------------------
# 2. Working with Dictionaries
person = {"name": "Alice", "age": 22}
print("Name:", person["name"])

# Add new key-value
person["city"] = "New York"

# Loop through dictionary
for key, value in person.items():
    print(f"{key}: {value}")

# --------------------------
# 3. Nested Lists (2D Lists)
matrix = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in matrix for item in sublist]
print("Flattened matrix:", flattened)
