import os
import re

# Directory containing the files
folder_path = 'C:/Users/giaco/OneDrive/Desktop/Università/Tesi_Master/Dataset_augmented'

# Regular expression pattern to extract person, weight, and attempt numbers from the filename
pattern = re.compile(r'data_neural_euler_acc_gyro_P(\d+)_W(\d+)_A(\d+)\.csv')

# Initialize a dictionary to keep track of the counts
file_structure = {}

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        person = int(match.group(1))
        weight = int(match.group(2))
        attempt = int(match.group(3))
        
        if person not in file_structure:
            file_structure[person] = {}
        if weight not in file_structure[person]:
            file_structure[person][weight] = []
        
        file_structure[person][weight].append(attempt)

# Verify the structure
expected_persons = 7279
expected_weights = 5
expected_attempts = 10

# Check if all persons are present
if len(file_structure) != expected_persons:
    print(f"Error: Expected {expected_persons} persons, but found {len(file_structure)}")
else:
    print("All persons are present")
error_p = 0
error_w = 0
error_a = 0
# Check each person for weights and attempts
for person in range(1, expected_persons + 1):
    if person not in file_structure:
        print(f"Error: Person {person} is missing")
        error_p += 1
        continue
    
    person_has_errors = False

    if len(file_structure[person]) != expected_weights:
        print(f"Error: Person {person} does not have {expected_weights} weights")
        person_has_errors = True
    
    for weight in range(1, expected_weights + 1):
        if weight not in file_structure[person]:
            print(f"Error: Person {person} is missing weight {weight}")
            error_w += 1
            person_has_errors = True
            continue
        
        if len(file_structure[person][weight]) != expected_attempts:
            print(f"Error: Person {person}, weight {weight} does not have {expected_attempts} attempts")
            person_has_errors = True
        else:
            for attempt in range(1, expected_attempts + 1):
                if attempt not in file_structure[person][weight]:
                    print(f"Error: Person {person}, weight {weight} is missing attempt {attempt}")
                    error_a += 1
                    person_has_errors = True
    
    if not person_has_errors:
        print(f"Person {person} check completed successfully")
    else:
        print(f"Person {person} check completed with errors")

print("Check completed")
print(error_p, error_w, error_a)
