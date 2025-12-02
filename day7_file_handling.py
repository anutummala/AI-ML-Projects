# day7_file_handling.py
# Python File Handling (CSV) Examples

import csv

# --------------------------
# 1. Write CSV file
with open("students.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Alice", 22, "New York"])
    writer.writerow(["Bob", 25, "London"])
    writer.writerow(["Bob1", 25, "London"])
    writer.writerow(["Bob2", 19, "Stockholm"])
    writer.writerow(["Bob3", 14, "Paris"])
    writer.writerow(["Bob4", 25, "London"])

# --------------------------
# 2. Read CSV file
print("\nReading CSV file:")
with open("students.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# --------------------------
# 3. Analyze CSV with DictReader
data = []
with open("students.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Count students older than 20 and find longest name
count_over_20 = sum(1 for row in data if int(row["Age"]) > 20)
longest_name = max(data, key=lambda x: len(x["Name"]))["Name"]

print("\nData:", data)
print("Number of students older than 20:", count_over_20)
print("Longest name:", longest_name)

# --------------------------
# 4. Students from London
london_students = [row for row in data if row["City"] == "London"]
print("\nStudents in London:")
for student in london_students:
    print(student)
