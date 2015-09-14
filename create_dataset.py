from sklearn.datasets import make_blobs
import csv

centers = [[-3, -3], [5,-5], [6,6]]
data_points, membership = make_blobs(n_samples=500, centers=centers)

# We only need the data_points now
with open('dataset.csv', 'w') as f:
    csv_writer = csv.writer(f)

    for point in data_points:
        csv_writer.writerow(point)