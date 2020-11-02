import numpy as np
import matplotlib.pyplot as plt

file_name = "log.lammps"

data_file = open(file_name, "r")
for i, line in enumerate(data_file):
    if "Step" in line:
        start_line = i
    if "Loop" in line:
        end_line = i
last_line = i
print(start_line, end_line, last_line)
data_file.close()

"""So it appears that each line is deleted from this as it is read. I have no idea why this is, but 
until I manage to fix that issue, I will close and reopen the file at this point"""

data = np.genfromtxt(
    fname=file_name,
    names=True,
    # delimiter=[8, 13, 13, 13, 13, 13],
    skip_header=start_line,
    skip_footer=last_line - end_line,
)

print(data.dtype.names)

plt.plot(data["Step"], data["TotEng"], label="Total Energy")
plt.plot(data["Step"], data["Temp"], label="Temperature")
plt.xlabel("Time Step")
plt.ylabel("Natural Units")
plt.legend()
plt.show()

