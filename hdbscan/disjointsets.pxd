from libc.stdint cimport uint32_t


cdef extern from "DisjointSets.h" nogil:
    cdef cppclass DisjointSets:
        DisjointSets(uint32_t) except +
        uint32_t size() const
        uint32_t find(uint32_t) const
        bint same(uint32_t, uint32_t) const
        uint32_t unite(uint32_t, uint32_t)
