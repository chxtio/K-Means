import numpy as np
import random
import matplotlib.pyplot as plt

data = np.load('AllSamples.npy')

def initial_point_idx(id, k, N):
	return np.random.RandomState(seed=(id+k)).permutation(N)[:k]

def init_point(data, idx):
    return data[idx,:]

def initial_S1(id, k_init_dict):
    i = int(id)%150
    random.seed(i+500)
    # Assign initial centers randomly
    for k in k_init_dict:
        init_idx = initial_point_idx(i,k,data.shape[0])
        k_init_dict[k] = init_point(data, init_idx)

    return k_init_dict

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
    plt.title("K ={}".format(k))
    plt.show()
    figure.savefig('s1_K-{}'.format(k))

    costFunctionJ = compute_loss_function(curr_centroids, clusters)
    print("\nCost function:", costFunctionJ)
    print("\nK-means algorithm converged after: {} iterations\n".format(iteration))

    return costFunctionJ

def compute_loss_function(centroids, clusters):
    cost = 0
    distances = []
    for i in range(centroids.shape[0]):
        sum = np.sum(np.square([np.linalg.norm(datapoint - centroids[i]) for datapoint in clusters[i]]))
        distances.append(sum)
    cost = np.sum(distances)

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
    figure.savefig('s1_cost_function')

def main():
    k_dict = dict.fromkeys([2, 3, 4, 5, 6, 7, 8, 9 ,10])
    initial_centroids_dict = initial_S1('6568', k_dict)
    x, y = zip(*data)
    for k in k_dict.keys():
        print("K =", k)
        k_dict[k] = kmeans(k, initial_centroids_dict)
    plot_cost_function(k_dict)

if __name__ == '__main__':
    main()
