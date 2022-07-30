import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skimage
from skimage import io


def load_image():
    filename = '../../data/raw/image.png'
    img = skimage.io.imread(filename)
    # print("Height:", img.shape[0], "Width:", img.shape[1], "Color channels:", img.shape[2])
    return img


def test():
    matrix = np.asarray([[1, 2, 3],
                         [4, 5, 6]])
    print("height", matrix.shape[0])
    print("width", matrix.shape[1])


def load_image_and_preprocess():
    image = load_image()
    rows = image.shape[0]
    cols = image.shape[1]
    # print(image, "Original")
    # Flatten the image
    image = image.reshape((image.shape[1] * image.shape[0], 3))
    # print("Height * Width:", image.shape[0], "Color Channels:", image.shape[1], )
    # print(image, "Reshape")

    return image, rows, cols


def get_labels_centroids(kmeans):
    labels = kmeans.labels_
    labels = list(labels)
    centroid = kmeans.cluster_centers_
    # print(centroid)
    return labels, centroid


def calculate_percentage(labels, centroid):
    percent = []
    for i in range(len(centroid)):
        j = labels.count(i)
        j = j / (len(labels))
        percent.append(j)
    return percent


def plot_pie(labels, centroid):
    percent = calculate_percentage(labels, centroid)
    plt.pie(percent, colors=np.array(centroid / 255),
            labels=np.arange(len(centroid)))
    plt.show()


def plot_compressed_image(compressed_image, k):
    io.imshow(compressed_image)
    io.show()
    filename = f'../../reports/figures/compressed_image_{k}.png'
    io.imsave(filename, compressed_image)


def recolor(centroid, labels, rows, cols, k):
    # Replace each pixel value with its nearby centroid
    compressed_image = centroid[labels]
    print("COMPRESSED", compressed_image)
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
    # Reshape the image to original dimension
    print("Compressed 2", compressed_image)
    print("Rows compresed", compressed_image.shape[0])
    print("Cols compresed", compressed_image.shape[1])

    print("Rows", rows)
    print("Cols", cols)
    compressed_image = compressed_image.reshape(rows, cols, 3)

    plot_compressed_image(compressed_image, k)
    plot_pie(labels, centroid)


def main():
    image, rows, cols = load_image_and_preprocess()
    k_list = [2, 4, 7, 10]

    for k in k_list:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(image)
        labels, centroid = get_labels_centroids(kmeans)
        print(f"Centroids for k = {k}:", centroid, "\n")
        recolor(centroid, labels, rows, cols, k)


if __name__ == '__main__':
    main()
