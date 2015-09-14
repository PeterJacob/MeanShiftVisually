from mean_shift import MeanShift
import matplotlib.pyplot as plt
import csv
import numpy as np

# Prepare mean shift model
bandwidth = 11
model = MeanShift(n_seeds=None, bandwidth=bandwidth)

# Read dataset from file
dataset = []
with open('dataset.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for point in csv_reader:
        float_point = [float(each) for each in point]
        dataset.append(float_point)
dataset = np.array(dataset)

# Fit model
fitted_centers = model.fit(dataset)

# Plot data
plt.scatter(dataset[:, 0], dataset[:, 1], marker='.', color='black', alpha=0.5)
plt.hold(True)

# Plot fitted cluster centers
plt.scatter(fitted_centers[:, 0], fitted_centers[:, 1], marker='*', color='green')

# Plot bandwidth ring around centers
fig = plt.gcf()
for center in fitted_centers:
    circle = plt.Circle(center, bandwidth, color='green', fill=False)
    fig.gca().add_artist(circle)

# Add labels
plt.title('Converged cluster centers with bandwidth=%s' % bandwidth)
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')
plt.axis('equal')
plt.axis([-10,10,-10,10])

plt.savefig('end_bandwidth_%s.png' % bandwidth)
plt.show()
