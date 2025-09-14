# Vector Module Documentation

This document provides detailed information about the Vector module, a dynamic array implementation in C for the DSL Standard Library. It supports generic data, dynamic resizing, insertion/removal at any position, sorting, and search algorithms.

## Table of Contents
- [Vector Structure](#structure)
- [Key Features](#key-features)
- [Initialization and Destruction](#initialization-and-destruction)
  - vector_init
  - vector_destroy
- [Capacity Management](#capacity-management)
  - vector_resize
  - vector_reserve
  - vector_set_capacity
  - vector_shrink
  - [Accessors](#accessors)
    - vector_get_size
    - vector_get_capacity
    - vector_get_element_size
    - vector_get_sorted
    - vector_is_empty
- [Element Access](#element-access)
  - vector_first
  - vector_last
  - vector_at
- [Insertion](#insertion)
  - vector_push
  - vector_push_front
  - vector_push_at
- [Removal](#removal)
  - vector_pop
  - vector_pop_front
  - vector_pop_at
  - vector_erase
  - vector_clear
- [Algorithms](#algorithms)
  - vector_foreach
  - vector_map
  - vector_find
  - vector_find_if
  - vector_contains
  - vector_bfind
  - vector_bfind_if
  - vector_bcontains
  - vector_swap
  - vector_qsort
  - vector_qsort_helper
  - vector_qsort_partition
- [Example Usage](#example-usage)

## Key Features
- vector_init / vector_destroy – setup and cleanup
- vector_push / vector_pop – add/remove elements
- vector_at / vector_first / vector_last – access elements
- vector_qsort / vector_find – algorithms

## Structure

```c++
typedef struct {
    void *data;           // Pointer to the array data
    unsigned int size;    // Current number of elements
    unsigned int capacity;// Allocated capacity of the vector
    unsigned int size_elem;// Size of each element in bytes
    char sorted;          // Flag indicating whether the vector is sorted
} Vector;

```

## Initialization and Destruction
```c++
vector_init(Vector* vec, unsigned int size_elem)
```
Initializes a vector for elements of size size_elem. Initial capacity is 1.<br>
Time Complexity: O(1)
```c++
vector_destroy(Vector* vec)
```
Frees memory used by the vector and resets all fields.<br>
Time Complexity: O(1)

## Capacity Management
```c++
vector_resize(Vector* vec)
```
Doubles the vector’s capacity. O(n)
```c++
vector_reserve(Vector* vec, unsigned int additional)
```
Ensures space for additional elements. O(n)
```c++
vector_set_capacity(Vector* vec, unsigned int new_capacity)
```
Explicitly sets vector capacity. O(n)
```c++
vector_shrink(Vector* vec)
```
Reduces capacity to match current size. O(n)

## Accessors:
```c++
unsigned int vector_get_size(Vector* vec);
```
Returns current number of elements
```c++
unsigned int vector_get_capacity(Vector* vec);
```
Returns allocated capacity
```c++
unsigned int vector_get_element_size(Vector* vec);
```
Returns size of each element in bytes
```c++
char vector_get_sorted(Vector* vec);
```
Returns sorted flag
```c++
char vector_is_empty(Vector* vec);
```
Returns 1 if empty, 0 otherwise
## Element Access
```c++
vector_first(Vector* vec, void* el) 
```
Copies first element into el. O(1)
```c++
vector_last(Vector* vec, void* el)
```
Copies last element into el. O(1)
```c++
vector_at(Vector* vec, unsigned int index, void* el) 
```
Copies element at index into el. O(1)
## Insertion
```c++
vector_push(Vector* vec, void* el)
```
Adds element to the end. O(n)

```c++
vector_push_front(Vector* vec, void* el)
```
Adds element to the start. O(n)

```c++
vector_push_at(Vector* vec, unsigned int index, void* el)
```
Inserts element at a given index. O(n)
## Removal
```c++
vector_pop(Vector* vec, void* el)
```
Removes last element and copies it into el. O(1)

```c++
vector_pop_front(Vector* vec, void* el)
```
Removes first element and copies it into el. O(n)

```c++
vector_pop_at(Vector* vec, unsigned int index, void* el)
```
Removes element at index and copies it into el. O(n)

```c++
vector_erase(Vector* vec, unsigned int index)
```
Removes element at index without returning it. O(n)

```c++
vector_clear(Vector* vec)
```
Removes all elements; retains allocated memory. O(n)
## Algorithms

```c++
vector_foreach(Vector* vec, void (*func)(void*, void*), void* user_data)
```
Applies func to each element with optional user data. O(n)

```c++
vector_map(Vector* vec, void (*func)(void*))
```
Applies func to each element in place. O(n)

```c++
vector_find(Vector* vec, void* el, int (*cmp)(const void*, const void*))
```
Returns index of el (linear search). O(n)

```c++
vector_find_if(Vector* vec, int (*predicate)(const void*))
```
Returns index of first element satisfying predicate. O(n)

```c++
vector_contains(Vector* vec, void* el, int (*cmp)(const void*, const void*))
```
Returns 1 if element exists, 0 otherwise. O(n)

```c++
vector_bfind(Vector* vec, void* el, int (*cmp)(const void*, const void*))
```
Returns index using binary search (requires sorted vector). O(log n)

```c++
vector_bfind_if(Vector* vec, int (*predicate)(const void*))
```
Returns index of first element satisfying predicate in sorted vector. O(log n)

```c++
vector_bcontains(Vector* vec, void* el, int (*cmp)(const void*, const void*))
```
Returns 1 if element exists using binary search. O(log n)

```c++
vector_swap(Vector* a, Vector* b)
```
Swaps two vectors (pointers only). O(1)

```c++
vector_qsort(Vector* vec, int (*cmp)(const void*, const void*))
```
Sorts vector using QuickSort. O(n log n) average, O(n²) worst

```c++
vector_qsort_helper(Vector* vec, int (*cmp)(const void*, const void*), int low, int high)
```
Internal helper for QuickSort.

```c++
vector_qsort_partition(Vector* vec, int (*cmp)(const void*, const void*), int low, int high)
```
Partitions vector for QuickSort.

## Example Usage
```c++
#include "DSL/dsl.h"
#include <stdio.h>

int main() {
    Vector vec;
    vector_init(&vec, sizeof(int));

    int x = 10;
    vector_push(&vec, &x);

    x = 20;
    vector_push(&vec, &x);

    int value;
    vector_at(&vec, 1, &value);
    printf("Second element: %d\n", value);

    vector_qsort(&vec, (int(*)(const void*, const void*)) strcmp); // Example sorting if elements are strings

    vector_destroy(&vec);
    return 0;
}
```