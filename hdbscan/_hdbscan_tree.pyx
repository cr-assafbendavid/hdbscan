# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# Tree handling (condensing, finding stable clusters) for hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

import logging
import numpy as np
cimport numpy as np

from cython.operator cimport preincrement, predecrement, postincrement

from .disjointsets cimport DisjointSets


cdef np.double_t INFTY = np.inf

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


cdef list bfs_from_hierarchy(const np.double_t[:, :] hierarchy, np.intp_t bfs_root):
    """
    Perform a breadth first search on a tree in scipy hclust format.
    """
    cdef Py_ssize_t i
    cdef list nodes = [bfs_root]
    cdef np.intp_t node, num_points = hierarchy.shape[0] + 1

    for node in nodes:
        if node >= num_points:
            i = <Py_ssize_t>(node - num_points)
            nodes.append(<np.intp_t>hierarchy[i, 0])
            nodes.append(<np.intp_t>hierarchy[i, 1])
    return nodes


cpdef np.ndarray condense_tree(const np.double_t[:, :] hierarchy, const np.double_t[:] sample_weight=None,
                               np.intp_t min_cluster_size=10):
    """Condense a tree according to a minimum cluster size. This is akin
    to the runt pruning procedure of Stuetzle. The result is a much simpler
    tree that is easier to visualize. We include extra information on the
    lambda value at which individual points depart clusters for later
    analysis and computation.

    Parameters
    ----------
    hierarchy : ndarray (n_samples, 4)
        A single linkage hierarchy in scipy.cluster.hierarchy format.

    min_cluster_size : int, optional (default 10)
        The minimum size of clusters to consider. Smaller "runt"
        clusters are pruned from the tree.

    Returns
    -------
    condensed_tree : numpy recarray
        Effectively an edgelist with a parent, child, lambda_val
        and child_weight in each row providing a tree structure.
    """
    cdef np.intp_t root = 2 * hierarchy.shape[0]
    cdef np.intp_t num_points = hierarchy.shape[0] + 1
    cdef np.intp_t next_label = num_points

    cdef list result_list = []
    cdef list node_list = bfs_from_hierarchy(hierarchy, root)

    cdef np.intp_t[::1] relabel = np.empty(root + 1, dtype=np.intp)
    relabel[root] = next_label

    cdef np.uint8_t[::1] ignore = np.zeros(len(node_list), dtype=bool)

    cdef Py_ssize_t i
    cdef np.intp_t node, sub_node
    cdef np.intp_t children[2]
    cdef np.double_t lambda_value
    cdef np.double_t weight[2]

    for node in node_list:
        if ignore[node] or node < num_points:
            continue

        lambda_value = hierarchy[node - num_points, 2]
        lambda_value = (1.0 / lambda_value) if lambda_value > 0.0 else INFTY
        for i in range(2):
            children[i] = <np.intp_t>hierarchy[node - num_points, i]
            weight[i] = hierarchy[children[i] - num_points, 3] if children[i] >= num_points else \
                1.0 if sample_weight is None else sample_weight[children[i]]

        if weight[0] >= min_cluster_size and weight[1] >= min_cluster_size:
            for i in range(2):
                relabel[children[i]] = preincrement(next_label)
                result_list.append((relabel[node], relabel[children[i]], lambda_value, weight[i]))
                if children[i] < num_points:
                    result_list.append((relabel[children[i]], children[i], INFTY, weight[i]))

        elif weight[0] >= min_cluster_size or weight[1] >= min_cluster_size:
            i = <Py_ssize_t>(weight[0] < min_cluster_size)
            relabel[children[i]] = relabel[node]
            if children[i] < num_points:
                result_list.append((relabel[children[i]], children[i], INFTY, weight[i]))

        for i in range(2):
            if weight[i] < min_cluster_size:
                for sub_node in bfs_from_hierarchy(hierarchy, children[i]):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value,
                                            1.0 if sample_weight is None else sample_weight[sub_node]))
                    ignore[sub_node] = True

    return np.array(result_list, dtype=[('parent', np.intp),
                                        ('child', np.intp),
                                        ('lambda_val', np.double),
                                        ('child_weight', np.double)])


cdef np.ndarray[np.double_t, ndim=1, mode='c'] compute_raw_stability(np.ndarray condensed_tree):
    cdef np.intp_t[:] parents = condensed_tree['parent']
    cdef np.intp_t[:] children = condensed_tree['child']
    cdef np.double_t[:] lambdas = condensed_tree['lambda_val']
    cdef np.double_t[:] sizes = condensed_tree['child_weight']

    cdef np.intp_t root = condensed_tree['parent'].min()
    cdef np.intp_t n_parents = max(condensed_tree['child'].max() - root + 1, 1)

    cdef Py_ssize_t i, j
    cdef np.double_t lambda_
    cdef np.double_t[::1] births = np.full(n_parents, np.inf, dtype=np.double)

    births[0] = 0.0
    for i in range(children.shape[0]):
        j = children[i] - root
        lambda_ = lambdas[i]
        if j > 0 and lambda_ < births[j]:
            births[j] = lambda_

    cdef np.ndarray[np.double_t, ndim=1, mode='c'] stability_arr = np.zeros(n_parents, dtype=np.double)
    cdef np.double_t[::1] stability = stability_arr

    for i in range(parents.shape[0]):
        j = parents[i] - root
        stability[j] += (lambdas[i] - births[j]) * sizes[i]

    return stability_arr


cpdef dict compute_stability(np.ndarray condensed_tree):
    cdef np.ndarray[np.double_t, ndim=1] raw_stability = compute_raw_stability(condensed_tree)
    return dict(enumerate(raw_stability, start=condensed_tree['parent'].min()))


cdef list all_descendants(const np.intp_t[:, ::1] links, np.intp_t node):
    cdef np.intp_t child
    cdef list descendants = [node]

    for node in descendants:
        child = links[node, 0]
        while child != -1:
            descendants.append(child)
            child = links[child, 1]
    return descendants[1:]


cdef np.ndarray[np.double_t, ndim=1] max_lambdas(np.ndarray tree, np.intp_t root):
    cdef Py_ssize_t i
    cdef np.intp_t[:] parents = tree['parent']
    cdef np.double_t[:] lambdas = tree['lambda_val']
    cdef np.ndarray[np.double_t, ndim=1] deaths_arr = np.zeros(tree['parent'].max() - root + 1, dtype=np.double)
    cdef np.double_t[::1] deaths = deaths_arr
    cdef np.intp_t parent
    cdef np.double_t lambda_

    for i in range(parents.shape[0]):
        parent = parents[i] - root
        lambda_ = lambdas[i]
        if lambda_ > deaths[parent]:
            deaths[parent] = lambda_
    return deaths_arr


cpdef np.ndarray[np.intp_t, ndim=1] labelling_at_cut(
        np.ndarray linkage,
        np.double_t cut,
        np.intp_t min_cluster_size):
    """Given a single linkage tree and a cut value, return the
    vector of cluster labels at that cut value. This is useful
    for Robust Single Linkage, and extracting DBSCAN results
    from a single HDBSCAN run.

    Parameters
    ----------
    linkage : ndarray (n_samples, 4)
        The single linkage tree in scipy.cluster.hierarchy format.

    cut : double
        The cut value at which to find clusters.

    min_cluster_size : int
        The minimum cluster size; clusters below this size at
        the cut will be considered noise.

    Returns
    -------
    labels : ndarray (n_samples,)
        The cluster labels for each point in the data set;
        a label of -1 denotes a noise assignment.
    """

    cdef np.uint32_t root, num_points, n, cluster, cluster_id
    cdef np.ndarray[np.intp_t, ndim=1] result_arr
    cdef np.ndarray[np.intp_t, ndim=1] unique_labels
    cdef np.ndarray[np.intp_t, ndim=1] cluster_size
    cdef np.intp_t[::1] result
    cdef DisjointSets *union_find

    root = 2 * linkage.shape[0]
    num_points = root // 2 + 1

    result_arr = np.empty(num_points, dtype=np.intp)
    result = result_arr

    union_find = new DisjointSets(root + 1)
    if union_find is NULL:
        raise MemoryError()

    cluster = num_points
    try:
        for row in linkage:
            if row[2] < cut:
                union_find.unite(<np.uint32_t>row[0], cluster)
                union_find.unite(<np.uint32_t>row[1], cluster)
            preincrement(cluster)

        cluster_size = np.zeros(cluster, dtype=np.intp)
        for n in range(num_points):
            cluster = union_find.find(n)
            cluster_size[cluster] += 1
            result[n] = cluster
    finally:
        del union_find

    cluster_label_map = {-1: -1}
    cluster_label = 0
    unique_labels = np.unique(result_arr)

    for cluster in unique_labels:
        if cluster_size[cluster] < min_cluster_size:
            cluster_label_map[cluster] = -1
        else:
            cluster_label_map[cluster] = cluster_label
            cluster_label += 1

    for n in range(num_points):
        result[n] = cluster_label_map[result[n]]

    return result_arr


cdef class Indexer(dict):
    def __getitem__(self, key):
        res = self.get(key)
        if res is None:
            res = len(self)
            self[key] = res
        return res

    def __array__(self, dtype=None):
        return np.fromiter(self.keys(), dtype=dtype, count=len(self))


cdef tuple do_labelling(
        np.ndarray tree,
        np.intp_t root,
        const np.uint8_t[::1] is_cluster,
        bint allow_single_cluster,
        bint match_reference_implementation):

    cdef np.uint32_t n
    cdef np.ndarray[np.intp_t, ndim=1] labels_arr = np.empty(root, dtype=np.intp)
    cdef np.intp_t[::1] labels = labels_arr
    cdef np.intp_t[:] parents = tree['parent']
    cdef np.intp_t[:] children = tree['child']
    cdef np.intp_t child, parent, cluster, label, c = 0
    cdef size_t n_clusters = 0
    cdef Indexer cluster_labels = Indexer()
    # DisjointSets will "find" the largest element (for our case, not in general),
    # so we must make the root the largest, hence the extra element
    cdef DisjointSets *union_find = new DisjointSets(<np.uint32_t>max(tree['child'].max(), tree['parent'].max()) + 2)
    if union_find is NULL:
        raise MemoryError()
    cdef np.intp_t dummy_root = union_find.size() - 1

    try:
        iterlogger = IterLogger(children.shape[0])
        for n in range(children.shape[0]):
            iterlogger.maybe_emit()
            child = children[n]
            # We've already asserted that children above the root (i.e. in "cluter_tree") are just a range,
            # starting from root. So we just access is_cluster in order (0 is the root itself, which is never a child).
            if child >= root and is_cluster[preincrement(c)]:
                preincrement(n_clusters)
            else:
                parent = parents[n]
                if parent == root:  # root is never a child
                    parent = dummy_root
                union_find.unite(<np.uint32_t>parent, <np.uint32_t>child)

        iterlogger = IterLogger(root)
        for n in range(root):
            iterlogger.maybe_emit()
            cluster = union_find.find(n)
            if cluster < root:
                label = -1
            elif cluster == dummy_root:
                if n_clusters == 1 and allow_single_cluster and \
                    tree['lambda_val'][tree['child'] == n] >= tree['lambda_val'][tree['parent'] == root].max():
                    label = cluster_labels[root]
                else:
                    label = -1
            else:
                if match_reference_implementation:
                    point_lambda = tree['lambda_val'][tree['child'] == n][0]
                    cluster_lambda = tree['lambda_val'][tree['child'] == cluster][0]
                    if point_lambda > cluster_lambda:
                        label = cluster_labels[cluster]
                    else:
                        label = -1
                else:
                    label = cluster_labels[cluster]
            labels[n] = label
    finally:
        del union_find
    return labels_arr, np.asarray(cluster_labels, dtype=np.intp)


cdef np.ndarray[np.double_t, ndim=1] get_probabilities(
        np.ndarray tree,
        np.intp_t root,
        const np.intp_t[::1] old_clusters,
        const np.intp_t[::1] labels):
    cdef np.ndarray[np.double_t, ndim=1] probs_arr = np.ones(labels.shape[0], dtype=np.double)
    cdef np.double_t[::1] probs = probs_arr
    cdef np.double_t[::1] deaths = max_lambdas(tree, root)
    cdef np.intp_t[:] nodes = tree['child']
    cdef np.double_t[:] lambdas = tree['lambda_val']
    cdef np.intp_t point, cluster_num
    cdef np.double_t max_lambda, prob
    cdef Py_ssize_t n
    cdef IterLogger iterlogger = IterLogger(nodes.shape[0])

    for n in range(nodes.shape[0]):
        iterlogger.maybe_emit()
        point = nodes[n]
        if point >= labels.shape[0]:
            continue
        cluster_num = labels[point]
        if cluster_num == -1:
            prob = 0
        else:
            max_lambda = deaths[old_clusters[cluster_num] - root]
            if max_lambda == 0 or not np.isfinite(lambdas[n]):
                prob = 1
            else:
                prob = min(lambdas[n], max_lambda) / max_lambda
        probs[point] = prob
    return probs_arr


cpdef np.ndarray[np.double_t, ndim=1] outlier_scores(np.ndarray tree):
    """Generate GLOSH outlier scores from a condensed tree.

    Parameters
    ----------
    tree : numpy recarray
        The condensed tree to generate GLOSH outlier scores from

    Returns
    -------
    outlier_scores : ndarray (n_samples,)
        Outlier scores for each sample point. The larger the score
        the more outlying the point.
    """

    cdef np.ndarray[np.double_t, ndim=1] result
    cdef np.ndarray[np.double_t, ndim=1] deaths
    cdef np.ndarray[np.double_t, ndim=1] lambda_array
    cdef np.ndarray[np.intp_t, ndim=1] child_array
    cdef np.ndarray[np.intp_t, ndim=1] parent_array
    cdef np.intp_t root_cluster
    cdef np.intp_t point
    cdef np.intp_t parent
    cdef np.intp_t cluster
    cdef np.double_t lambda_max

    child_array = tree['child']
    parent_array = tree['parent']
    lambda_array = tree['lambda_val']

    root_cluster = parent_array.min()
    deaths = max_lambdas(tree, root_cluster)
    result = np.zeros(root_cluster, dtype=np.double)

    topological_sort_order = np.argsort(parent_array)
    # topologically_sorted_tree = tree[topological_sort_order]

    for n in topological_sort_order:
        cluster = child_array[n]
        if cluster < root_cluster:
            break

        parent = parent_array[n]
        if deaths[cluster - root_cluster] > deaths[parent - root_cluster]:
            deaths[parent - root_cluster] = deaths[cluster - root_cluster]

    for n in range(tree.shape[0]):
        point = child_array[n]
        if point >= root_cluster:
            continue

        cluster = parent_array[n]
        lambda_max = deaths[cluster - root_cluster]

        if lambda_max == 0.0 or not np.isfinite(lambda_array[n]):
            result[point] = 0.0
        else:
            result[point] = (lambda_max - lambda_array[n]) / lambda_max

    return result


cdef np.ndarray[np.double_t, ndim=1] get_stability_scores(const np.intp_t[::1] labels,
                                                          const np.intp_t[::1] old_clusters,
                                                          const np.double_t[::1] stability,
                                                          np.double_t max_lambda,
                                                          np.intp_t root):
    cdef Py_ssize_t i
    cdef np.intp_t cluster_size, label
    cdef np.ndarray[np.double_t, ndim=1] stability_score_arr = np.empty(old_clusters.shape[0], dtype=np.double)
    cdef np.double_t[::1] stability_score = stability_score_arr
    cdef np.intp_t[::1] cluster_sizes = np.zeros(old_clusters.shape[0], dtype=np.intp)
    cdef IterLogger iterlogger = IterLogger(old_clusters.shape[0])

    for i in range(labels.shape[0]):
        label = labels[i]
        if label != -1:
            preincrement(cluster_sizes[label])

    for i in range(old_clusters.shape[0]):
        iterlogger.maybe_emit()
        cluster_size = cluster_sizes[i]
        if np.isinf(max_lambda) or max_lambda == 0.0 or cluster_size < 2:
            stability_score[i] = 1.0
        else:
            stability_score[i] = stability[old_clusters[i] - root] / (cluster_size * max_lambda)

    return stability_score_arr


cpdef list recurse_leaf_dfs(np.ndarray cluster_tree, np.intp_t current_node):
    children = cluster_tree[cluster_tree['parent'] == current_node]['child']
    if len(children) == 0:
        return [current_node]
    return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])


cpdef list get_cluster_tree_leaves(np.ndarray cluster_tree):
    return recurse_leaf_dfs(cluster_tree, cluster_tree['parent'].min()) if cluster_tree.shape[0] > 0 else []


cdef np.intp_t traverse_upwards(const np.intp_t[:] parents, const np.double_t[:] lambdas, np.intp_t root,
                                np.double_t cluster_selection_lambda, np.intp_t node,
                                bint allow_single_cluster) nogil:
    cdef np.intp_t parent

    while True:
        parent = parents[node - 1] - root
        if parent == 0:
            break
        if lambdas[parent - 1] < cluster_selection_lambda:
            return parent
        node = parent

    if allow_single_cluster:
        return 0  # the root
    # return node closest to root
    return node


cdef np.ndarray[np.intp_t, ndim=1] epsilon_search(const np.intp_t[::1] nodes,
                                                  np.ndarray cluster_tree,
                                                  const np.intp_t[:, ::1] cluster_links,
                                                  np.intp_t root,
                                                  np.double_t cluster_selection_epsilon,
                                                  bint allow_single_cluster):
    cdef set selected_clusters = set(), processed = set()
    cdef np.double_t cluster_selection_lambda = 1.0 / cluster_selection_epsilon
    cdef np.intp_t[:] parents = cluster_tree['parent']
    cdef np.double_t[:] lambdas = cluster_tree['lambda_val']
    cdef np.intp_t node, epsilon_child
    cdef Py_ssize_t i

    for i in range(nodes.shape[0]):
        node = nodes[i]
        if lambdas[node - 1] > cluster_selection_lambda:
            if node not in processed:
                epsilon_child = traverse_upwards(parents, lambdas, root, cluster_selection_lambda, node,
                                                 allow_single_cluster)
                selected_clusters.add(epsilon_child)
                processed.update(all_descendants(cluster_links, epsilon_child))
        else:
            selected_clusters.add(node)
    return np.fromiter(selected_clusters, dtype=np.intp, count=len(selected_clusters))


cpdef tuple get_clusters(np.ndarray tree,
                         cluster_selection_method='eom',
                         allow_single_cluster=False,
                         match_reference_implementation=False,
                         cluster_selection_epsilon=0.0):
    """Given a tree and stability dict, produce the cluster labels
    (and probabilities) for a flat clustering based on the chosen
    cluster selection method.

    Parameters
    ----------
    tree : numpy recarray
        The condensed tree to extract flat clusters from

    cluster_selection_method : string, optional (default 'eom')
        The method of selecting clusters. The default is the
        Excess of Mass algorithm specified by 'eom'. The alternate
        option is 'leaf'.

    allow_single_cluster : boolean, optional (default False)
        Whether to allow a single cluster to be selected by the
        Excess of Mass algorithm.

    match_reference_implementation : boolean, optional (default False)
        Whether to match the reference implementation in how to handle
        certain edge cases.

    cluster_selection_epsilon: float, optional (default 0.0)
        A distance threshold for cluster splits.

    Returns
    -------
    labels : ndarray (n_samples,)
        An integer array of cluster labels, with -1 denoting noise.

    probabilities : ndarray (n_samples,)
        The cluster membership strength of each sample.

    stabilities : ndarray (n_clusters,)
        The cluster coherence strengths of each cluster.
    """
    # Assume clusters are ordered by numeric id equivalent to
    # a topological sort of the tree; This is valid given the
    # current implementation above, so don't change that ... or
    # if you do, change this accordingly!
    logger.info("computing stability scores")
    cdef np.double_t[::1] stability = compute_raw_stability(tree)
    cdef np.intp_t root = tree['parent'].min()

    cdef np.ndarray cluster_tree = tree[tree['child'] >= root]
    assert_cluster_children_validity(cluster_tree['child'], root)

    logger.info("preparing cluster tree links")
    cdef np.intp_t[:, ::1] cluster_links = prepare_tree_links(cluster_tree, True)

    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] is_cluster_arr = np.empty(stability.shape[0], dtype=bool)
    cdef np.uint8_t[::1] is_cluster = is_cluster_arr

    logger.info(f"selecting clusters using {cluster_selection_method} method")
    if cluster_selection_method == 'eom':
        select_clusters_eom(cluster_links, stability, allow_single_cluster, is_cluster)
        if cluster_selection_epsilon != 0.0:
            logger.info(f"correcting nodes selection for epsilon={cluster_selection_epsilon}")
            selected_clusters = epsilon_search(is_cluster_arr.nonzero()[0], cluster_tree, cluster_links, root,
                                               cluster_selection_epsilon, allow_single_cluster)
            is_cluster[:] = False
            is_cluster_arr[selected_clusters] = True

    elif cluster_selection_method == 'leaf':
        selected_clusters = np.asarray(get_cluster_tree_leaves(cluster_tree), dtype=np.intp)
        if len(selected_clusters) == 0:
            selected_clusters = 0  # the root
        elif cluster_selection_epsilon != 0.0:
            logger.info(f"correcting nodes selection for epsilon={cluster_selection_epsilon}")
            selected_clusters = epsilon_search(selected_clusters - root, cluster_tree, cluster_links, root,
                                               cluster_selection_epsilon, allow_single_cluster)
        else:
            selected_clusters -= root
        is_cluster[:] = False
        is_cluster_arr[selected_clusters] = True
    else:
        raise ValueError('Invalid Cluster Selection Method: %s\n'
                         'Should be one of: "eom", "leaf"\n')

    cdef np.ndarray[np.intp_t, ndim=1] labels, old_clusters
    cdef np.ndarray[np.double_t, ndim=1] probs, stabilities
    logger.info("labeling clusters")
    labels, old_clusters = do_labelling(tree, root, is_cluster, allow_single_cluster, match_reference_implementation)
    logger.info("computing cluster probabilities")
    probs = get_probabilities(tree, root, old_clusters, labels)
    logger.info("computing cluster stability scores")
    stabilities = get_stability_scores(labels, old_clusters, stability, tree['lambda_val'].max(), root)
    return labels, probs, stabilities


cdef void assert_cluster_children_validity(const np.intp_t[:] children, np.intp_t root):
    cdef Py_ssize_t i
    for i in range(children.shape[0]):
        if children[i] != preincrement(root):
            raise ValueError("unexpected cluster_tree structure")


cdef np.ndarray[np.intp_t, ndim=2, mode='c'] prepare_tree_links(np.ndarray tree, bint partial):
    # `links` has an extra row because the root is not part of the tree
    # if building for a partial tree, links[0] is the root
    # column 0 is the first child, 1 is the next sibling
    cdef np.ndarray[np.intp_t, ndim=2, mode='c'] links_arr = np.full((tree.size + 1, 2), -1, dtype=np.intp)
    cdef np.intp_t[:, ::1] links = links_arr
    cdef np.intp_t[:] nodes = tree['child']
    cdef np.intp_t[:] parents = tree['parent']
    cdef np.intp_t[::1] parent, sib
    cdef IterLogger iterlogger = IterLogger(parents.shape[0])
    cdef np.intp_t p, node, root
    cdef Py_ssize_t i

    if partial:
        root = tree['parent'].min() if parents.shape[0] else -1
    for i in range(parents.shape[0]):
        iterlogger.maybe_emit()
        p = parents[i]
        if partial:
            p -= root
            node = i + 1
        else:
            node = nodes[i]
        parent = links[p]
        if parent[0] == -1:
            parent[0] = node
        else:
            sib = links[parent[0]]
            while sib[1] != -1:
                sib = links[sib[1]]
            sib[1] = node
    return links_arr


cdef void select_clusters_eom(const np.intp_t[:, ::1] links,
                              np.double_t[::1] stability,
                              bint allow_single_cluster,
                              np.uint8_t[::1] is_cluster) nogil:
    if links.shape[0] != stability.shape[0] or stability.shape[0] != is_cluster.shape[0]:
        with gil:
            raise ValueError("shapes of links, stability and is_cluster don't match")

    cdef np.intp_t node, child, sib, n_iter
    cdef np.double_t subtree_stability

    n_iter = stability.shape[0]
    if not allow_single_cluster:
        is_cluster[0] = False
        predecrement(n_iter)

    with gil:
        iterlogger = IterLogger(n_iter)

    for node in range(stability.shape[0] - 1, -1 if allow_single_cluster else 0, -1):
        iterlogger.maybe_emit()
        child = links[node, 0]
        if child == -1:
            subtree_stability = 0.0
        else:
            subtree_stability = stability[child]
            sib = links[child, 1]
            while sib != -1:
                subtree_stability += stability[sib]
                sib = links[sib, 1]
        is_cluster[node] = False
        if subtree_stability > stability[node]:
            stability[node] = subtree_stability
        else:
            zero_subtree(links, is_cluster, node)
            is_cluster[node] = True


cdef void zero_subtree(const np.intp_t[:, ::1] links, np.uint8_t[::1] is_cluster, np.intp_t node) nogil:
    if is_cluster[node]:
        # The subtree below a True node is already entirely False
        is_cluster[node] = False
        return
    cdef np.intp_t child, sib
    child = links[node, 0]
    if child == -1:
        return
    zero_subtree(links, is_cluster, child)
    sib = links[child, 1]
    while sib != -1:
        zero_subtree(links, is_cluster, sib)
        sib = links[sib, 1]


cdef class IterLogger:
    cdef int current, n_iter, last
    cdef float factor, next

    def __init__(self, size_t n_iter, float factor=2.0):
        self.n_iter = <int>n_iter
        self.last = self.n_iter - 1
        self.factor = factor
        self.current = -1
        self.next = 0

    cdef void maybe_emit(self) nogil:
        preincrement(self.current)
        if self.current >= self.next or self.current == self.last:
            with gil:
                logger.debug(f"--> working on iteration {self.current} out of {self.n_iter}")
            self.next = self.current * self.factor
