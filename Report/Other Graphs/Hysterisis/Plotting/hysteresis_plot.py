import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 13})  # for figures to go into latex at halfwidth

exp_data = pd.read_csv("exp.csv")
cont_data = pd.read_csv("cont.csv")
print(exp_data.shape)

plt.plot(exp_data["vol_frac"], exp_data["order"], "rx", label="Expansion")
plt.plot(cont_data["vol_frac"], cont_data["order"], "bx", label="Contraction")
plt.ylabel("Order Parameter")
plt.xlabel("Volume Fraction")
plt.legend()
plt.savefig("order_vs_volfrac.png")
plt.show()

exp_data2 = pd.read_csv("exp2.csv")
cont_data2 = pd.read_csv("cont2.csv")
print(exp_data.shape)

plt.plot(exp_data2["vol_frac"], exp_data2["order"], "rx", label="Expansion")
plt.plot(cont_data2["vol_frac"], cont_data2["order"], "bx", label="Contraction")
plt.ylabel("Order Parameter")
plt.xlabel("Volume Fraction")
plt.legend()
plt.savefig("order_vs_volfrac2.png")
plt.show()

