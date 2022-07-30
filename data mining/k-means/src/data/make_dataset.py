import skimage
import matplotlib.pyplot as plt


def load_image():
    filename = '../../data/raw/image.png'
    img = skimage.io.imread(filename)
    return img


def plot_image(img):
    skimage.io.imshow(img)
    plt.show()


def main():
    img = load_image()
    # plot_image(img)
    print("Height:", img.shape[0], "Width:", img.shape[1], "Color channels:", img.shape[2])



if __name__ == '__main__':
    main()
