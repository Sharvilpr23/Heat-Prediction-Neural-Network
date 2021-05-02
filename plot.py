import matplotlib.pyplot as plt

try:
    fin = open("test.txt", 'r')
except:
    print("Failed to open the file")
    quit()

lines = fin.readlines()

x = []
y = []

for line in lines:
    line = line.strip().split()
    x.append(int(line[0]))
    y.append(float(line[1]))

plt.xlabel("Run")
plt.ylabel("Correctly Identified %")
plt.ylim(85.0, 100.0)
plt.scatter(x, y, color = 'red')
plt.title("Efficiency over 100 Runs")
plt.show()