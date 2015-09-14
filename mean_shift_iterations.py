from mean_shift import MeanShift
import matplotlib.pyplot as plt
import csv
import numpy as np

# Prepare mean shift model
bandwidth = 2
model = MeanShift(n_seeds=1, bandwidth=bandwidth)

# Read dataset from file
dataset = []
with open('dataset.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for point in csv_reader:
        float_point = [float(each) for each in point]
        dataset.append(float_point)
dataset = np.array(dataset)

# Fit model
model.fit(dataset)
cluster_center_history = model.cluster_center_history

for iteration, cluster_centers in enumerate(cluster_center_history):
    # Plot data
    plt.scatter(dataset[:, 0], dataset[:, 1], marker='.', color='black', alpha=0.5)
    plt.hold(True)

    # Plot fitted cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', color='green')

    # Plot bandwidth ring around centers
    fig = plt.gcf()
    for center in cluster_centers:
        #print('{},{}'.format(center[0], center[1]))
        circle = plt.Circle(center, bandwidth, color='green', fill=False)
        fig.gca().add_artist(circle)

    # Add labels
    plt.title('Cluster center at iteration %s' % iteration)
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.axis('equal')
    plt.axis([-10,10,-10,10])

    plt.savefig('interation_single_%s.png' % iteration)
    plt.close()