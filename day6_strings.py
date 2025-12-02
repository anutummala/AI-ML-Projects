# day6_strings.py
# Python String Manipulation Examples

# --------------------------
# 1. Count vowels in a string
text = input("Enter a string: ")
vowel_count = 0
for char in text.lower():
    if char in 'aeiou':
        vowel_count += 1
print("Number of vowels:", vowel_count)

# --------------------------
# 2. Reverse a string
def reverse_string(input_str):
    return input_str[::-1]

reversed_text = reverse_string(text)
print(f"Original: {text}")
print(f"Reversed: {reversed_text}")

# --------------------------
# 3. Word frequency count
text_sample = "This is another sample text. Another text."
words = text_sample.lower().replace('.', '').split()
word_freq = {}
for word in words:
    word_freq[word] = word_freq.get(word, 0) + 1
print("Word frequencies:", word_freq)

# --------------------------
# 4. String trimming, lowercasing, and replacement
input_string = " Hello AI World "
trimmed = input_string.strip()
lowercase = trimmed.lower()
final_string = lowercase.replace("ai", "Artificial Intelligence")
print("Final string:", final_string)
