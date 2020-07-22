

import numpy as np, sklearn, umap
import sklearn.metrics
from sklearn import preprocessing


def aknn(nbrs_arr, labels, thresholds, distinct_labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
    """
    Apply AKNN rule for a query point, given its list of nearest neighbors.

    Parameters
    ----------
    nbrs_arr: array di dimensione (n_neighbors)
        Indici dei `n_neighbors` nearest neighbors neldataset.
    labels: array di dimensione (n_samples)
        Labels del dataset.

    thresholds: array di dimensione (n_neighbors)
        Bias thresholds per divverenti dimensioni di neighborhood.
    Returns
    -------
    pred_label: string
        Predizione della label di AKNN.
    first_admissible_ndx: int
        n-1, dove AKNN sceglie la dimensione del neighborhood pari a n.

    fracs_labels: array di dimensione (n_labels, n_neighbors)
        Frazione di ogni label in circonferenze di diversa dimensione per ogni neighborhood.
    """
    query_nbrs = labels[nbrs_arr]
    mtr = np.stack([query_nbrs == i for i in distinct_labels])
    rngarr = np.arange(len(nbrs_arr)) + 1
    fracs_labels = np.cumsum(mtr, axis=1) / rngarr
    biases = fracs_labels - 1.0 / len(distinct_labels)
    numlabels_predicted = np.sum(biases > thresholds, axis=0)
    admissible_ndces = np.where(numlabels_predicted > 0)[0]
    first_admissible_ndx = admissible_ndces[0] if len(admissible_ndces) > 0 else len(nbrs_arr)
    pred_label = '?' if first_admissible_ndx == len(nbrs_arr) else distinct_labels[np.argmax(biases[:,
                                                                                             first_admissible_ndx])]
    return (pred_label, first_admissible_ndx, fracs_labels)


def predict_nn_rule(nbr_list_sorted, labels, log_complexity=1.0,
                    distinct_labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
    """
    Data una matrice di nearest neighbors ordinati per ogni punto, ritorna la predizione della classe fatta da AKNN and e le dimensioni adattive del neighborhood.

    Parameters
    ----------
    nbr_list_sorted: array di dimensione (n_samples, n_neighbors)
        Indici dei `n_neighbors` nearest neighbors nel dataset, per ogni punto.
    labels: array di dimensione (n_samples)
        Labels del dataset.

    log_complexity: float
        Il parametro di confidenza "A".
    Returns
    -------
    pred_labels: array di dimensione (n_samples)
        Predizioni delle label di AKNN sul dataset.
    adaptive_ks: array di dimensione (n_samples)
        Dimensioni dei AKNN neighborhood sul dataset.
    """
    pred_labels = []
    adaptive_ks = []
    thresholds = log_complexity / np.sqrt(np.arange(nbr_list_sorted.shape[1]) + 1)
    distinct_labels = np.unique(labels)
    for i in range(nbr_list_sorted.shape[0]):
        (pred_label, adaptive_k_ndx, _) = aknn(nbr_list_sorted[i, :], labels, thresholds)
        pred_labels.append(pred_label)
        adaptive_ks.append(adaptive_k_ndx + 1)
    return np.array(pred_labels), np.array(adaptive_ks)


def calc_nbrs_exact(raw_data, k=1000):
    """
    Calcola la lista dei `k` esatti nearest neighbors Euclidei per ogni punto.

    Parameters
    ----------
    raw_data: array di dimensione (n_samples, n_features)
        Dataset di input.
    Returns
    -------
    nbr_list_sorted: array di dimensionee (n_samples, n_neighbors)
        Indici dei `n_neighbors` nearest neighbors nel dataset, per ogni punto.
    """
    a = sklearn.metrics.pairwise_distances(raw_data)
    nbr_list_sorted = np.argsort(a, axis=1)[:, 1:]
    return nbr_list_sorted[:, :k]


def knn_rule(nbr_list_sorted, labels, k=10):
    toret = []
    for i in range(nbr_list_sorted.shape[0]):
        uq = np.unique(labels[nbr_list_sorted[i, :k]], return_counts=True)
        toret.append(uq[0][np.argmax(uq[1])])
    return np.array(toret)