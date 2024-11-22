import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_simplex(topic_weights, topic_labels=None, point_labels=None):
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    if topic_labels:
        for idx, coord in enumerate(triangle):
            plt.text(coord[0], coord[1], topic_labels[idx], fontsize=12, ha='center', va='center')
    for simplex_edge in [[0, 1], [1, 2], [2, 0]]:
        plt.plot(triangle[simplex_edge, 0], triangle[simplex_edge, 1], 'k-', lw=1)
    simplex_points = np.dot(topic_weights, triangle)
    plt.scatter(simplex_points[:, 0], simplex_points[:, 1], c='blue', s=50, alpha=0.7)
    if point_labels:
        for idx, point in enumerate(simplex_points):
            plt.text(point[0], point[1], point_labels[idx], fontsize=9, ha='center', va='center')
    plt.axis('off')
    plt.show()


def plot_3d_simplex(topic_weights, topic_labels=None, point_labels=None):
    tetrahedron = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1/3, 1/3, 1/3]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for edge in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
        ax.plot(*zip(*tetrahedron[edge]), color='black', lw=1)
    simplex_points = np.dot(topic_weights, tetrahedron)
    ax.scatter(simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2], c='blue', s=50, alpha=0.7)
    if topic_labels:
        for idx, coord in enumerate(tetrahedron):
            ax.text(coord[0], coord[1], coord[2], topic_labels[idx], fontsize=12, ha='center', va='center')
    if point_labels:
        for idx, point in enumerate(simplex_points):
            ax.text(point[0], point[1], point[2], point_labels[idx], fontsize=9, ha='center', va='center')
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def normalize_weights(weights):
    weights = np.array(weights)
    return weights / weights.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    example_2d_weights = normalize_weights([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.5, 0.3, 0.2]])
    example_3d_weights = normalize_weights([[0.5, 0.3, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3], [0.1, 0.6, 0.2, 0.1]])

    print("Plotting 2D simplex...")
    plot_2d_simplex(example_2d_weights, topic_labels=["A", "B", "C"], point_labels=["P1", "P2", "P3"])

    print("Plotting 3D simplex...")
    plot_3d_simplex(example_3d_weights, topic_labels=["X", "Y", "Z", "W"], point_labels=["D1", "D2", "D3"])

