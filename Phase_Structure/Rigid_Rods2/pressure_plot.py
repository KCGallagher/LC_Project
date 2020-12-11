import numpy as np
import matplotlib.pyplot as plt

file_name = "log.lammps"

data_file = open(file_name, "r")
for i, line in enumerate(data_file):
    if "Step" in line:
        start_line = i  # start of final data readout
    if "Loop" in line:
        end_line = i  # end of final data readout
last_line = i  # last line number in file
print(start_line, end_line, last_line)
data_file.close()
# This considers the last data output only

"""So it appears that each line is deleted from this as it is read. I have no idea why this is, but 
until I manage to fix that issue, I will close and reopen the file if necessary at this point"""

data = np.genfromtxt(
    fname=file_name,
    names=True,
    # delimiter=[8, 13, 13, 13, 13, 13],
    skip_header=start_line,
    skip_footer=last_line - end_line,
)

print(data.dtype.names)

plt.plot(data["Step"], data["Press"], label="Pressure")
plt.xlabel("Time Step")
plt.ylabel("Natural Units")
plt.legend()
plt.show()

