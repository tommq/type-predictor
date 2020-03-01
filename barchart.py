import matplotlib.pyplot as plt

sizes = [10, 20, 30, 50, 70, 90]
mlp = [40, 50, 56, 63, 80, 81]
svm = [61, 61, 66, 72, 75, 78]
lr = [58, 62, 72, 78, 81, 82]

fig, ax = plt.subplots()
ax.plot(sizes, mlp, label="mlp", marker='o')
ax.plot(sizes, svm, label="svm", marker='o')
ax.plot(sizes, lr, label="lr", marker='o')
plt.xlabel('Number of training samples per class in train dataset')
plt.ylabel('Average accuracy in 5-fold CV')
ax.legend()

plt.show()
