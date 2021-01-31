import numpy as np

file_root = "output_T_0.5_time_"

# this gives the number of blank lines encountered by each Loop line

time_range = range(0, 500, 100)

for t in time_range:  # interate over dump files
    data_file = open(file_root + str(t) + ".dump", "r")
    extract_data = False  # start of file doesn't contain particle values

    for i, line in enumerate(data_file):
        if "ITEM: ATOMS" in line:  # to start reading data
            extract_data = True
            continue  # don't attempt to read this line
        if extract_data:
            particle_values = []
            for t in line.split():  # separate by whitespace
                try:
                    particle_values.append(float(t))
                except ValueError:
                    pass  # any non-floats in this line are ignored
