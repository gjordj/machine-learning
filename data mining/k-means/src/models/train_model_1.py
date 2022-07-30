import skimage
from skimage import io
import numpy as np


class KMean:
    def __init__(self, k, max_iter=50):
        """

        Args:
            k: The number of clusters to form as well as the number of centroids to generate.
            max_iter:  Maximum number of iterations of the k-means algorithm for a single run.
        """

        self.k = k
        self.centroids = None
        self.max_iter = max_iter

    def initialize_centroids(self, points, k):
        """ Returns

        Args:
            points: image to work with
            k: The number of clusters to form as well as the number of centroids to generate.

        Returns: k centroids from the initial points.


        """
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[:k]

    # def initialize_centroids(self, points, k):
    #     """ Returns
    #
    #         Args:
    #             points: image to work with
    #             k: The number of clusters to form as well as the number of centroids to generate.
    #
    #         Returns: k centroids from the initial points.
    #
    #
    #         """        centroids = points.copy()
    #     np.random.shuffle(centroids)
    #     if k == 2:
    #         centroids = [[0, 0, 0],
    #                      [0.2, 0.2, 0.2]]
    #         centroids = np.asarray(centroids)
    #     elif k == 4:
    #         centroids = [[0, 0, 0],
    #                      [0.1, 0.1, 0.1],
    #                      [0.2, 0.2, 0.2],
    #                      [0.3, 0.3, 0.3]]
    #         centroids = np.asarray(centroids)
    #
    #     elif k == 7:
    #         centroids = [[0, 0, 0],
    #                      [0.1, 0.1, 0.1],
    #                      [0.2, 0.2, 0.2],
    #                      [0.3, 0.3, 0.3],
    #                      [0.4, 0.4, 0.4],
    #                      [0.5, 0.5, 0.5],
    #                      [0.6, 0.6, 0.6]]
    #         centroids = np.asarray(centroids)
    #
    #     elif k == 10:
    #         centroids = [[0, 0, 0],
    #                      [0.1, 0.1, 0.1],
    #                      [0.2, 0.2, 0.2],
    #                      [0.3, 0.3, 0.3],
    #                      [0.4, 0.4, 0.4],
    #                      [0.5, 0.5, 0.5],
    #                      [0.6, 0.6, 0.6],
    #                      [0.7, 0.7, 0.7],
    #                      [0.8, 0.8, 0.8],
    #                      [0.9, 0.9, 0.9]]
    #         centroids = np.asarray(centroids)
    #
    #     return centroids

    def closest_centroid(self, points, centroids):
        """

        Args:
            points: data to work with
            centroids: array with the centroids.

        Returns: array containing the index to the nearest centroid for each point

        """
        diff = points - centroids[:, np.newaxis]
        distances = np.sqrt((diff ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def move_centroids(self, points, closest, centroids):
        """

        Args:
            points: image to work with.
            closest: array containing the index to the nearest centroid for each point
            centroids: array with the centroids.

        Returns: array with the new centroids assigned from the points closest to them

        """

        new_list = []
        for k in range(centroids.shape[0]):
            new_list.append(points[closest == k].mean(axis=0))
        return np.array(new_list)

    def fit(self, points):
        """

        Args:
            points: image to work with.

        Returns: labels based on closest center

        """
        self.centroids = self.initialize_centroids(points, self.k)
        # print("Initialized centroids:\n", self.centroids)
        centroids = None
        for i in range(self.max_iter):
            closest = self.predict(points)
            self.centroids = self.move_centroids(points, closest, self.centroids)
        # print("Moved centroids:\n", self.centroids)
        return closest

    def predict(self, points):
        """

        Args:
            points: image to work with

        Returns: array containing the index to the nearest centroid for each point.

        """
        closest = self.closest_centroid(points, self.centroids)
        return closest


def load_image():
    """

    Returns: image to work with in the right format.

    """
    filename = '../../data/raw/image.png'
    img = skimage.io.imread(filename)
    # print("Height:", img.shape[0], "Width:", img.shape[1], "Color channels:", img.shape[2])
    return img


def load_image_and_preprocess():
    """

    Returns:
        - reshaped image in the right format
        - number of rows of the formatted image
        - number of columns of the formatted image

    """
    image = load_image()
    rows = image.shape[0]
    cols = image.shape[1]
    # print(image, "Original")
    # Flatten the image
    image = image.reshape((image.shape[1] * image.shape[0], 3))
    # print("Height * Width:", image.shape[0], "Color Channels:", image.shape[1], )
    # print(image, "Reshape")

    return image, rows, cols


def plot_compressed_image(compressed_image, k):
    """

    Args:
        compressed_image: image compressed with k number of colors
        k: number of clusters.

    Returns: plot

    """
    io.imshow(compressed_image)
    io.show()
    filename = f'../../reports/figures/compressed_image_{k}.png'
    io.imsave(filename, compressed_image)


def plot_recolor(k_new_colors, rows, cols, k):
    """

    Args:
        compressed_image: image compressed with k number of colors
        rows: number of rows of the formatted original image.
        cols: number of columns of the formatted original image.
        k: number of clusters.

    Returns: plot.

    """
    # Replace each pixel value with its nearby centroid
    k_new_colors = np.clip(k_new_colors.astype('uint8'), 0, 255)
    # Reshape the image to original dimension
    k_new_colors = k_new_colors.reshape(rows, cols, 3)
    plot_compressed_image(k_new_colors, k)


def recolor(kmeans, k, labels):
    dict = {2: np.array([[60, 179, 113],
                         [0, 191, 255]]),
            4: np.array([[60, 179, 113],
                         [0, 191, 255],
                         [255, 255, 0],
                         [255, 0, 0]]),
            7: np.array([[60, 179, 113],
                         [0, 191, 255],
                         [255, 255, 0],
                         [255, 0, 0],
                         [0, 0, 0],
                         [169, 169, 169],
                         [255, 140, 0]]),
            10: np.array([[60, 179, 113],
                          [0, 191, 255],
                          [255, 255, 0],
                          [255, 0, 0],
                          [0, 0, 0],
                          [169, 169, 169],
                          [255, 140, 0],
                          [128, 0, 128],
                          [128, 0, 128],
                          [255, 255, 255]])
            }

    kmeans.centroids = dict[k]
    compressed_image = kmeans.centroids[labels]
    return compressed_image


def sse_calculation(k, labels, image):
    """

    Args:
        k: number of clusters
        labels: array containing the index to the nearest centroid for each point
        image: original formatted image

    Returns:
        clusterwise_sse: The clusterwise SSE

    """
    cluster_centers = [image[labels == i].mean(axis=0) for i in range(k)]
    # print( kmeans.centroids == cluster_centers)
    # print("centers", cluster_centers)
    clusterwise_sse = [0] * k
    for point, label in zip(image, labels):
        clusterwise_sse[label] += np.square(point - cluster_centers[label]).sum()
    print("SSE:\n", clusterwise_sse)


def image_compression_and_recolor():
    """

    Returns: The plot of an image formatted in k different colors using k-means clustering.

    """
    k_list = [2, 4, 7, 10]

    image, rows, cols = load_image_and_preprocess()

    for k in k_list:
        kmeans = KMean(k)
        labels = kmeans.fit(image)
        sse_calculation(k, labels, image)
        k_new_colors = recolor(kmeans, k, labels)
        plot_recolor(k_new_colors, rows, cols, k)


def main():
    image_compression_and_recolor()


if __name__ == '__main__':
    main()
