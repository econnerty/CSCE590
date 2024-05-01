import numpy as np
import matplotlib.pyplot as plt
import heapq

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params


def test_adam():
    def loss(X):
        return np.sum((X - 1) ** 2), 2 * (X - 1)

    num_params = 50
    X = np.random.randn(num_params)
    optimizer = AdamOptimizer(lr=0.1)
    history = np.zeros((200, num_params))

    for i in range(200):
        l, grad = loss(X)
        X = optimizer.update(X, grad)
        history[i] = X

    plt.figure(figsize=(10, 6))
    for j in range(num_params):
        plt.plot(history[:, j], label=f'Param {j+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('ADAM Optimization')
    plt.legend(loc='best', ncol=5)
    plt.savefig('adam.pdf')
    plt.close()


def dijkstra(graph, start):
    num_nodes = len(graph)
    dist = [float('inf')] * num_nodes
    dist[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_dist, node = heapq.heappop(priority_queue)

        if current_dist > dist[node]:
            continue

        for neighbor, weight in enumerate(graph[node]):
            if weight > 0:
                distance = current_dist + weight

                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    return dist


def test_dijkstra():
    graph = np.random.randint(0, 20, size=(10, 10))
    np.fill_diagonal(graph, 0)

    start_node = 0
    distances = dijkstra(graph, start_node)

    plt.bar(range(len(distances)), distances)
    plt.xlabel('Node')
    plt.ylabel('Distance from Node 0')
    plt.title('Dijkstra Algorithm')
    plt.legend([f"Start Node: {start_node}"], loc='upper right')
    plt.savefig('dijkstra.pdf')
    plt.close()


def kmeans(X, k, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    initial_centroids = centroids.copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels, initial_centroids


def test_kmeans():
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(loc=[0, 0], scale=2, size=(300, 2)),
        np.random.normal(loc=[10, 10], scale=2, size=(300, 2)),
        np.random.normal(loc=[20, 0], scale=2, size=(300, 2)),
        np.random.normal(loc=[10, -10], scale=2, size=(300, 2))
    ])

    k = 4
    centroids, labels, initial_centroids = kmeans(X, k)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, label='Data Points')
    plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='blue', marker='o', s=100, label='Initial Centroids')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Final Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend(loc='best')
    plt.savefig('kmeans.pdf')
    plt.close()


if __name__ == "__main__":
    test_adam()
    test_dijkstra()
    test_kmeans()
