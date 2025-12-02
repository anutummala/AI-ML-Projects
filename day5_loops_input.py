# day5_loops_input.py
# Python Loops, Range, and Input Examples

# --------------------------
# 1. Multiplication Table
for i in range(1, 6):  # Rows
    for j in range(1, 6):  # Columns
        print(f"{i*j:4}", end="")
    print()  # Newline after each row

# --------------------------
# 2. Squares of Even Numbers (List Comprehension)
numbers = list(range(1, 21))
even_squares = [x**2 for x in numbers if x % 2 == 0]
print("Original numbers:", numbers)
print("Squares of even numbers:", even_squares)

# --------------------------
# 3. Flattening a Nested List (2D list)
matrix = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in matrix for item in sublist]
print("Flattened matrix:", flattened)

# --------------------------
# 4. Extract letters from list of words
words = ["ryuuoo", "uiojhkll", "pidfjgjjdss"]
letters = [letter for word in words for letter in word]
print("All letters:", letters)
