import networkx
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix)
from matplotlib import pylab as pl
import matplotlib.pyplot as plt

from abp.graphml.datasets.dataset import Dataset

ds = Dataset('data/adjacency.csv', 'data/features.csv', 200)

ds.print_summary()

all_posts = [post.to_nx_graph() for post in ds.data.values()]
all_labels = [post.label for post in ds.data.values()]
print('Total number of posts: ', len(all_posts))

regular_posts = [post for (post, label) in zip(all_posts, all_labels) if label == 0]
ad_posts = [post for (post, label) in zip(all_posts, all_labels) if label == 1]

print('Number of regular posts: ', len(regular_posts))
print('Number of ads: ', len(ad_posts))

import grakel

POSTS_COUNT = 150

sel_posts = all_posts[:POSTS_COUNT]
sel_labels = all_labels[:POSTS_COUNT]

g_posts = grakel.graph_from_networkx(sel_posts)
kernel = grakel.ShortestPath(with_labels=False, normalize=True, verbose=True)

K_all = kernel.fit_transform(g_posts)

K_posts = [post for (post, label) in zip(K_all, sel_labels) if label == 0]
K_ads = [post for (post, label) in zip(K_all, sel_labels) if label == 1]

K_train, K_test, y_train, y_test = train_test_split(K_all, sel_labels, test_size=0.2, random_state=42)
