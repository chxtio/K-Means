import random
import numpy as np
import matplotlib.pyplot as plt
data = np.load('AllSamples.npy')

def initial_point_idx2(id,k, N):
    random.seed((id+k))
    return random.randint(0,N-1)

def initial_S2(id, k_init_dict):
    i = int(id)%150
    random.seed(i+800)
    for k in k_init_dict:
        init_idx2 = initial_point_idx2(i, k, data.shape[0])
        k_init_dict[k] = data[init_idx2,:]

    return k_init_dict

def initial_centroids(id, k_init_dict):
    k_initial_points = initial_S2(id, k_init_dict)
    # Select new points furthest from previous centers
    for k in k_initial_points:
        skip = np.array([centroid for centroid in k_initial_points[k]])
        for addCenter in range(k - 1):
            maxDistance = 0
            candidate = np.zeros(shape=2)
            for datapoint in data:
                if datapoint in skip:
                    continue
                distance = [np.linalg.norm(datapoint - centroid) for centroid in k_initial_points[k]]
                avgDistance = np.average(distance)
                if avgDistance > maxDistance:
                    maxDistance = avgDistance
                    candidate = datapoint
            skip = np.vstack((skip, candidate))
            k_initial_points[k] = np.vstack((k_initial_points[k],candidate))
    plotInitialCentroids(k_initial_points)

    return k_initial_points

def plotInitialCentroids(clusters_dict):
    x,y = zip(*data)
    for k, centroids in clusters_dict.items():
        figure = plt.figure()
        cInitX, cInitY = zip(*(centroids))
        plt.scatter(x,y, s=10, color='black')
        plt.scatter(cInitX, cInitY, marker='*', color='red', s=50)
        plt.title("K = {}".format(k))
        plt.show()
        figure.savefig('s2_init_centroids_K-{}'.format(k))

def kmeans(k, centroids):
    iteration = 0
    old_centroids = None
    curr_centroids = centroids[k]
    print("Initialization centroids\n", centroids[k], "\n")
    # Repeat until convergence or stop at max iterations
    while iteration < 50 and (not np.array_equal(old_centroids, curr_centroids)):
        iteration += 1
        centroids[k] = curr_centroids
        # Cluster assignment- assign samples to closest centroids
        clusters = classify(k, centroids)
        old_centroids = curr_centroids
        # Calculate new centroids from clusters
        curr_centroids = recompute(centroids[k], clusters)
    print("Final centroids")
    print(curr_centroids)
    plt.style.use('ggplot')
    figure = plt.figure()
    for centroid, samples in clusters.items():
        clusters[centroid] = np.array(samples)
        x, y = zip(*samples)
        plt.scatter(x,y, s=10)
    cluster_x, cluster_y = zip(*(centroids[k]))
    plt.scatter(cluster_x, cluster_y, marker='*', s=60, color='red')
    plt.title("K = {}".format(k))
    plt.show()
    figure.savefig('s2_kmeans_K-{}'.format(k))
    costFunctionJ = compute_loss_function(curr_centroids, clusters)

    return costFunctionJ

def compute_loss_function(centroids, clusters):
    cost = 0
    distances = []
    for i in range(centroids.shape[0]):
        sum = np.sum(np.square([np.linalg.norm(datapoint - centroids[i]) for datapoint in clusters[i]]))
        distances.append(sum)
    cost = np.sum(distances)
    print("\nCost function:", cost)

    return cost

def classify(k, centroids):
    classes = {}
    for i in range(k):
        classes[i] = []
    for sample in data:
        distance = [np.linalg.norm(sample - centroid) for centroid in centroids[k]]
        classIndex = distance.index(min(distance))
        classes.setdefault(classIndex, []).append(sample)

    return classes

def recompute(old_centroids, clusters):
    centroids_new = np.zeros(shape=(len(old_centroids), 2))
    for i in range(len(clusters)):
        centroids_new[i] = np.average(clusters[i], axis=0)

    return centroids_new

def plot_cost_function(plot_function_dict):
    x,y = zip(*sorted(plot_function_dict.items()))
    plt.style.use('default')
    figure = plt.figure()
    plt.plot(x, y, 'b', marker='o')
    plt.title("K Means Algorithm")
    plt.xlabel("K (no. of clusters)")
    plt.ylabel("Cost function J")
    plt.show
    figure.savefig('s2_cost_function')

def main():
    k_dict = dict.fromkeys([2, 3, 4, 5, 6, 7, 8, 9 ,10])
    initial_centroids_dict = initial_centroids('6568', k_dict)
    for k in k_dict.keys():
        print("K =", k)
        k_dict[k] = kmeans(k, initial_centroids_dict)
    plot_cost_function(k_dict)

if __name__ == '__main__':
    main()
