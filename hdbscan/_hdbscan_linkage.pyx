# cython: boundscheck=False
# cython: nonecheck=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes, Steve Astels
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX
from cython.operator cimport postincrement

from .dist_metrics cimport DistanceMetric


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core(
                               np.ndarray[np.double_t,
                                          ndim=2] distance_matrix):

    cdef np.ndarray[np.intp_t, ndim=1] node_labels
    cdef np.ndarray[np.intp_t, ndim=1] current_labels
    cdef np.ndarray[np.double_t, ndim=1] current_distances
    cdef np.ndarray[np.double_t, ndim=1] left
    cdef np.ndarray[np.double_t, ndim=1] right
    cdef np.ndarray[np.double_t, ndim=2] result

    cdef np.ndarray label_filter

    cdef np.intp_t current_node
    cdef np.intp_t new_node_index
    cdef np.intp_t new_node
    cdef np.intp_t i

    result = np.zeros((distance_matrix.shape[0] - 1, 3))
    node_labels = np.arange(distance_matrix.shape[0], dtype=np.intp)
    current_node = 0
    current_distances = np.infty * np.ones(distance_matrix.shape[0])
    current_labels = node_labels
    for i in range(1, node_labels.shape[0]):
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]
        left = current_distances[label_filter]
        right = distance_matrix[current_node][current_labels]
        current_distances = np.where(left < right, left, right)

        new_node_index = np.argmin(current_distances)
        new_node = current_labels[new_node_index]
        result[i - 1, 0] = <double> current_node
        result[i - 1, 1] = <double> new_node
        result[i - 1, 2] = current_distances[new_node_index]
        current_node = new_node

    return result


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core_vector(
        np.ndarray[np.double_t, ndim=2, mode='c'] raw_data,
        np.ndarray[np.double_t, ndim=1, mode='c'] core_distances,
        DistanceMetric dist_metric,
        np.double_t alpha=1.0):

    # Add a comment
    cdef np.ndarray[np.double_t, ndim=1] current_distances_arr
    cdef np.ndarray[np.int8_t, ndim=1] in_tree_arr
    cdef np.ndarray[np.double_t, ndim=2] result_arr

    cdef np.double_t * current_distances
    cdef np.double_t * current_core_distances
    cdef np.double_t * raw_data_ptr
    cdef np.int8_t * in_tree
    cdef np.double_t[:, ::1] raw_data_view
    cdef np.double_t[:, ::1] result

    cdef np.ndarray label_filter

    cdef np.intp_t current_node
    cdef np.intp_t new_node
    cdef np.intp_t i
    cdef np.intp_t j
    cdef np.intp_t dim
    cdef np.intp_t num_features

    cdef double current_node_core_distance
    cdef double right_value
    cdef double left_value
    cdef double core_value
    cdef double new_distance

    dim = raw_data.shape[0]
    num_features = raw_data.shape[1]

    raw_data_view = (<np.double_t[:raw_data.shape[0], :raw_data.shape[1]:1]> (
        <np.double_t *> raw_data.data))
    raw_data_ptr = (<np.double_t *> &raw_data_view[0, 0])

    result_arr = np.zeros((dim - 1, 3))
    in_tree_arr = np.zeros(dim, dtype=np.int8)
    current_node = 0
    current_distances_arr = np.infty * np.ones(dim)

    result = (<np.double_t[:dim - 1, :3:1]> (<np.double_t *> result_arr.data))
    in_tree = (<np.int8_t *> in_tree_arr.data)
    current_distances = (<np.double_t *> current_distances_arr.data)
    current_core_distances = (<np.double_t *> core_distances.data)

    for i in range(1, dim):

        in_tree[current_node] = 1

        current_node_core_distance = current_core_distances[current_node]

        new_distance = DBL_MAX
        new_node = 0

        for j in range(dim):
            if in_tree[j]:
                continue

            right_value = current_distances[j]
            left_value = dist_metric.dist(&raw_data_ptr[num_features *
                                                        current_node],
                                          &raw_data_ptr[num_features * j],
                                          num_features)

            if alpha != 1.0:
                left_value /= alpha

            core_value = core_distances[j]
            if (current_node_core_distance > right_value or
                    core_value > right_value or
                    left_value > right_value):
                if right_value < new_distance:
                    new_distance = right_value
                    new_node = j
                continue

            if core_value > current_node_core_distance:
                if core_value > left_value:
                    left_value = core_value
            else:
                if current_node_core_distance > left_value:
                    left_value = current_node_core_distance

            if left_value < right_value:
                current_distances[j] = left_value
                if left_value < new_distance:
                    new_distance = left_value
                    new_node = j
            else:
                if right_value < new_distance:
                    new_distance = right_value
                    new_node = j

        result[i - 1, 0] = <double>current_node
        result[i - 1, 1] = <double>new_node
        result[i - 1, 2] = new_distance
        current_node = new_node

    return result_arr


cdef class UnionFind:
    cdef np.intp_t n_elements
    cdef np.intp_t[::1] parent
    cdef np.double_t[::1] size

    def __init__(self, np.intp_t n_elements):
        self.n_elements = n_elements
        self.parent = np.full(2 * n_elements - 1, -1, dtype=np.intp, order='C')
        self.size = np.empty(self.parent.shape[0], dtype=np.double)
        self.size[:n_elements] = 1

    cdef np.double_t _unify(self, np.intp_t node1, np.intp_t node2):
        cdef np.double_t union_size = self.size[node1] + self.size[node2]
        self.size[postincrement(self.n_elements)] = union_size
        return union_size

    cdef (np.intp_t, np.intp_t, np.double_t) union(self, np.intp_t node1, np.intp_t node2):
        cdef np.intp_t root1 = self._find(node1)
        cdef np.intp_t root2 = self._find(node2)
        return root1, root2, self._unify(root1, root2)

    cdef np.intp_t _find(self, np.intp_t node) nogil:
        cdef np.intp_t parent = self.parent[node]
        while parent != -1:
            self.parent[node] = self.n_elements
            node = parent
            parent = self.parent[node]
        self.parent[node] = self.n_elements
        return node


cdef class WeightedUnionFind(UnionFind):
    def __init__(self, np.double_t[:] weights):
        super().__init__(weights.shape[0])
        self.size[:self.n_elements] = weights


def label(const np.double_t[:, :] mst, np.double_t[:] sample_weight=None):
    cdef Py_ssize_t i
    cdef np.intp_t left_root, right_root
    cdef np.double_t size
    result_arr = np.empty((mst.shape[0], mst.shape[1] + 1), dtype=np.double, order='C')
    cdef np.double_t[:, ::1] result = result_arr
    cdef UnionFind uf

    if sample_weight is None:
        uf = UnionFind(mst.shape[0] + 1)
    else:
        uf = WeightedUnionFind(sample_weight)
    for i in range(mst.shape[0]):
        left_root, right_root, size = uf.union(<np.intp_t>mst[i, 0], <np.intp_t>mst[i, 1])
        result[i, 0] = left_root
        result[i, 1] = right_root
        result[i, 3] = size
    result[:, 2] = mst[:, 2]
    return result_arr


cpdef np.ndarray[np.double_t, ndim=2] single_linkage(distance_matrix):

    cdef np.ndarray[np.double_t, ndim=2] hierarchy
    cdef np.ndarray[np.double_t, ndim=2] for_labelling

    hierarchy = mst_linkage_core(distance_matrix)
    for_labelling = hierarchy[np.argsort(hierarchy.T[2]), :]

    return label(for_labelling)
