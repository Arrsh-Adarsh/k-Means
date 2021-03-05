#############################################################
#                                                           #
#       | /    |\ /| |'''   /\   |\  | ;------              #
#       |-     | ' | |--   /  \  | \ | |_____.              #
#       | \    |   | |... /----\ |  \| ._____|              #
#                                                           #
#                                                           #
#############################################################


# importing libraries

from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import style
from algo import K_Means

style.use('ggplot')


# main(driver) function
def main():
    # loading dataset from skit learn
    iris = datasets.load_iris()
    X = iris.data[:, 2:4]
    y = iris.target
    km = K_Means(3)
    km.fit(X)

    colors = ["r", "g", "c", "b", "k"]

    # plotting graph for the predicted data
    for cluster in km.classes:
        color = colors[cluster]
        for features in km.classes[cluster]:
            plt.scatter(features[0], features[1], color=color, s=30)

    for centroid in km.centroids:
        plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="*", color='k')

    plt.xlabel('Petal Length', fontsize=18)
    plt.ylabel('Petal Width', fontsize=18)
    plt.show()


# program execution start from here
if __name__ == '__main__':
    main()
