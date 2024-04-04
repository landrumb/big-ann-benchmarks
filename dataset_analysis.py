# %%
from scipy.sparse import csr_matrix
from benchmark import datasets, dataset_io
import numpy as np
import matplotlib.pyplot as plt
from math import log2
import time

# %%
yfcc = datasets.YFCC100MDataset()

# %%
dataset_metadata = yfcc.get_dataset_metadata()
print(dataset_metadata)

# %%
dataset_metadata.shape

# %%
# compute the number of filters per data point
n_filters = dataset_metadata.getnnz(axis=1)
print(n_filters)

# %%
# compute the number of data points associated with each label
n_assoc = dataset_metadata.getnnz(axis=0)
print("n_assoc", n_assoc.shape, n_assoc.min(), n_assoc.max())

# %%
# create a list of sets where each element contains the labels associated with a data point
labels_per_data_point = [set(dataset_metadata.indices[dataset_metadata.indptr[i]
                             :dataset_metadata.indptr[i+1]]) for i in range(dataset_metadata.shape[0])]
print(labels_per_data_point[0])

# %%
# plot histogram of labels per datapoint
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.hist(n_filters, bins=100)
# ax.set_title("Histogram of labels per datapoint")
# ax.set_xlabel("Number of labels")
# ax.set_ylabel("Number of datapoints")
# # ax.set_yscale("log")
# plt.show()

# %%
# plot histogram of labels per datapoint
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.hist(n_filters, bins=100)
# ax.set_title("Histogram of labels per datapoint (log scale)")
# ax.set_xlabel("Number of labels")
# ax.set_ylabel("Number of datapoints")
# ax.set_yscale("log")
# plt.show()

# %%
print(f"min: {min(n_filters)}, max: {max(n_filters)}, mean: {np.mean(n_filters)}, median: {np.median(n_filters)}")

# %%
# thresholds = [1000, 500, 250, 100, 50, 25, 10, 5, 2, 1]
thresholds = [10**(e/4) for e in range(0, 16)]
qualifying = []
for threshold in thresholds:
    qualifying.append(np.sum(np.array(n_filters) >= threshold))
output = ""
for t, n in zip(thresholds, qualifying):
    output += f"{n} ({n / len(n_filters) * 100:.3f}%) points with at least {t:,} labels\n"
print(output)

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(thresholds, qualifying, "o-")
# ax.set_xscale("log")
# # ax.set_yscale("log")
# ax.set_xlabel("Threshold")
# ax.set_ylabel("Number of points")
# ax.set_title("Number of points with at least $n$ labels")
# plt.show()

# %%
quantiles = [0.025, 0.05, 0.125, 0.25, 0.5, 0.75, 0.875, 0.95, 0.99, 0.999, 1.]
for q, n in zip(quantiles, np.quantile(n_filters, quantiles)):
    print(f"{q:.3f}: {n:.0f}")

# %% [markdown]
# ## Points Per Label

# %%
# plot histogram of points per label
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.hist(n_assoc, bins=100)
# ax.set_title("Histogram of datapoints per label")
# ax.set_xlabel("Number of datapoints")
# ax.set_ylabel("Number of labels")
# # ax.set_yscale("log")
# plt.show()

# %%
# plot histogram of points per label
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.hist(n_assoc, bins=100)
# ax.set_title("Histogram of datapoints per label (log scale)")
# ax.set_xlabel("Number of datapoints")
# ax.set_ylabel("Number of labels")
# ax.set_yscale("log")
# plt.show()

# %%
print(f"min: {min(n_assoc):,} ({min(n_assoc) / dataset_metadata.shape[0] * 100:.3f}%),\nmax: {max(n_assoc):,} ({max(n_assoc) / dataset_metadata.shape[0] * 100:.3f}%),\nmean: {np.mean(n_assoc):.2f} ({np.mean(n_assoc) / dataset_metadata.shape[0] * 100:.3f}%),\nmedian: {np.median(n_assoc):,} ({np.median(n_assoc) / dataset_metadata.shape[0] * 100:.3f}%)")

# %%
k = 25
top_k = np.sort(n_assoc, axis=0)[-k:]
top_k = top_k[::-1]

for i, s in enumerate(top_k):
    print(f"{i+1}. {s / dataset_metadata.shape[0] * 100:.3f}%\n")

# %%
thresholds = [10**e for e in range(7)][::-1]
qualifying = []
for threshold in thresholds:
    qualifying.append(np.sum(np.array(n_assoc) >= threshold))
output = ""
for t, n in zip(thresholds, qualifying):
    output += f"{n} ({n / len(n_assoc) * 100:.3f}%) labels with at least {t:,} associations\n"
print(output)

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(thresholds, qualifying, "o-")
# ax.set_xscale("log")
# ax.set_title("Number of labels with at least n associations")
# ax.set_xlabel("Threshold")
# ax.set_ylabel("Number of labels")
# plt.show()

# %%
quantiles = [0.025, 0.05, 0.125, 0.25, 0.5, 0.75, 0.875, 0.95, 0.99, 0.999, 1.]
for q, n in zip(quantiles, np.quantile(n_assoc, quantiles)):
    print(f"{q:.3f}: {n:.0f}")

# %% [markdown]
# # Public Queries

# %%
query_md = yfcc.get_queries_metadata()
print(query_md)

# %%
n_criteria = query_md.getnnz(axis=1)
print(f"1 filter: {np.sum(n_criteria == 1):,}\n2 filters: {np.sum(n_criteria == 2):,}\n3 filters: {np.sum(n_criteria == 3)}")

# %%
queries = []
for i in range(100000):
    queries.append(query_md[i, :].nonzero()[1])

# %%
single_queries = np.concatenate([arr for arr in queries if arr.size == 1])
double_queries = np.vstack([arr for arr in queries if arr.size == 2])

# %%
sq_select = n_assoc[single_queries] / 10**7
sq_select

# %%
np.max(sq_select)

# %%
dq_select = n_assoc[double_queries[:, 0]] * \
    n_assoc[double_queries[:, 1]] / 10**14
dq_select

# %%
# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# ax[0].hist(sq_select, bins=100, range=(0, 0.2))
# ax[0].set_title("Selectivities of Single Filter Queries")
# ax[0].set_xlabel("Selectivity")
# ax[0].set_ylabel("Number of Queries")
# # ax[0].set_xscale("log")

# ax[1].hist(dq_select, bins=100, range=(0, 0.2))
# ax[1].set_title("Selectivities of Double Filter Queries (projected)")
# ax[0].set_xlabel("Selectivity")
# ax[0].set_ylabel("Number of Queries")

# plt.tight_layout()

# %% [markdown]
# building a query co-occurence matrix

# %%
# dot_prod = dataset_metadata.dot(dataset_metadata.T)

# %%
# from scipy.sparse import save_npz

# save_npz("metadata_dot_product.npz", dot_prod)

# %%
dataset_metadata.count_nonzero()

# %% [markdown]
# ## Chains of Label Sets
# How many unique chains of weakly increasing sets of labels partially ordered by inclusion are there?

# %%


def unique(lst):
    output = []
    for x in lst:
        if x not in output:
            output.append(x)
    return output


tab = "\t"
SUBSET_SIZE = 10_000_000


def rec_unique(lst):
    n = len(lst)
    if n > 10_000:
        print(f"{tab * round(log2(SUBSET_SIZE / n))}Splitting list of length {n:,} into two lists of length {n//2:,} and {n-n//2:,} ({time.strftime('%H:%M:%S')})")
    if n <= 100:
        return unique(lst)
    else:
        output = []
        b = rec_unique(lst[:n//2])
        for a in rec_unique(lst[n//2:]):
            if a not in b:
                output.append(a)
        return b + output


unique_label_sets = rec_unique(labels_per_data_point[:SUBSET_SIZE])

UNIQUE_SETS_PATH = "unique_label_sets.txt"
with open(UNIQUE_SETS_PATH, "w") as f:
    for s in unique_label_sets:
        for x in s:
            f.write(f"{x} ")
        f.write("\n")

# %%
# {1, 2, 3, 4}.issuperset({1, 2, 3, 4})
