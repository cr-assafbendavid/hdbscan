from libc.stdint cimport uint32_t

# TODO:
#  - Incorporate code from https://github.com/wjakob/dset
#  - Change id1 < id2 to id1 > id2 in `unite` and other methods, so that
#    the representative found is consistently the largest element.


cdef extern from "DisjointSets.h" nogil:
    cdef cppclass DisjointSets:
        DisjointSets(uint32_t) except +
        uint32_t size() const
        uint32_t find(uint32_t) const
        bint same(uint32_t, uint32_t) const
        uint32_t unite(uint32_t, uint32_t)
