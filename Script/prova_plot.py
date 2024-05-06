import csv
import matplotlib.pyplot as plt

csv_file_path = r'C:\Users\giaco\OneDrive\Desktop\Università\Tesi_Master\GitHub\Dataset\P1\W1\A1\emg\emg_data.csv'  # Replace with the path to your CSV file

# Lists to store the values from the first column
values = []

# Open the CSV file for reading
with open(csv_file_path, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Append the value of the first column (index 0) to the list
        values.append(-float(row[0]))  # Convert the value to float if necessary

# Plot the values
plt.plot(values)
plt.xlabel('Index')  # Set label for x-axis
plt.ylabel('Value')  # Set label for y-axis
plt.title('Values from First Column')  # Set title of the plot
plt.grid(True)  # Enable grid
plt.show()  # Show the plot
