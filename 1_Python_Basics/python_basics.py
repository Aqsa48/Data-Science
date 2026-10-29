"""
Python Basics for Data Science
================================
Covers: variables, data types, control flow, functions, list comprehensions, and OOP.
"""

# ---------------------------------------------------------
# 1. Variables & Data Types
# ---------------------------------------------------------
integer_val = 42
float_val = 3.14
string_val = "Data Science"
bool_val = True
none_val = None

print("=== Variables & Data Types ===")
print(f"int:    {integer_val}  ({type(integer_val).__name__})")
print(f"float:  {float_val}  ({type(float_val).__name__})")
print(f"str:    {string_val}  ({type(string_val).__name__})")
print(f"bool:   {bool_val}  ({type(bool_val).__name__})")

# ---------------------------------------------------------
# 2. Collections
# ---------------------------------------------------------
print("\n=== Collections ===")

# List – ordered, mutable
fruits = ["apple", "banana", "cherry"]
fruits.append("date")
print("List:", fruits)

# Tuple – ordered, immutable
coordinates = (10.0, 20.0)
print("Tuple:", coordinates)

# Set – unordered, unique values
unique_nums = {1, 2, 2, 3, 3, 3}
print("Set:", unique_nums)

# Dictionary – key-value pairs
person = {"name": "Alice", "age": 30, "job": "Data Scientist"}
print("Dict:", person)

# ---------------------------------------------------------
# 3. Control Flow
# ---------------------------------------------------------
print("\n=== Control Flow ===")

score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
print(f"Score {score} → Grade {grade}")

for fruit in fruits[:3]:
    print(f"  Fruit: {fruit}")

count = 0
while count < 3:
    print(f"  Count: {count}")
    count += 1

# ---------------------------------------------------------
# 4. Functions
# ---------------------------------------------------------
print("\n=== Functions ===")


def greet(name, greeting="Hello"):
    """Return a greeting string."""
    return f"{greeting}, {name}!"


print(greet("Alice"))
print(greet("Bob", greeting="Hi"))


def calculate_bmi(weight_kg, height_m):
    """Calculate and return BMI."""
    return round(weight_kg / (height_m ** 2), 2)


print(f"BMI: {calculate_bmi(70, 1.75)}")

# Lambda function
square = lambda x: x ** 2
print("Squares:", list(map(square, range(1, 6))))

# ---------------------------------------------------------
# 5. List Comprehensions
# ---------------------------------------------------------
print("\n=== List Comprehensions ===")

squares = [x ** 2 for x in range(1, 11)]
print("Squares 1-10:", squares)

even_squares = [x ** 2 for x in range(1, 11) if x % 2 == 0]
print("Even squares:", even_squares)

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["data", "science", "python"]}
print("Word lengths:", word_lengths)

# ---------------------------------------------------------
# 6. Exception Handling
# ---------------------------------------------------------
print("\n=== Exception Handling ===")


def safe_divide(a, b):
    """Divide a by b, returning None on error."""
    try:
        return a / b
    except ZeroDivisionError:
        print("  Error: division by zero")
        return None
    except TypeError as e:
        print(f"  TypeError: {e}")
        return None


print("10 / 2 =", safe_divide(10, 2))
print("10 / 0 =", safe_divide(10, 0))

# ---------------------------------------------------------
# 7. Object-Oriented Programming
# ---------------------------------------------------------
print("\n=== Object-Oriented Programming ===")


class DataPoint:
    """Represents a single data observation."""

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __repr__(self):
        return f"DataPoint(feature={self.feature}, label={self.label!r})"

    def is_positive(self):
        """Return True if the label is positive (1)."""
        return self.label == 1


class LabeledDataset:
    """A simple labeled dataset."""

    def __init__(self):
        self.data = []

    def add(self, feature, label):
        self.data.append(DataPoint(feature, label))

    def positives(self):
        return [dp for dp in self.data if dp.is_positive()]

    def __len__(self):
        return len(self.data)


dataset = LabeledDataset()
for feat, lbl in [(1.2, 1), (3.4, 0), (5.6, 1), (0.9, 0)]:
    dataset.add(feat, lbl)

print(f"Dataset size: {len(dataset)}")
print("Positive samples:", dataset.positives())

print("\nDone! Python basics covered successfully.")
