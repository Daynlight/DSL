# Vector Module Documentation

This document provides detailed information about the `Vector` module, a dynamic array implementation in C for the DSL Standard Library.

---

## Vector Structure

```c
typedef struct {
    void *data;          // Pointer to array data
    unsigned int size;   // Number of elements in the vector
    unsigned int capacity; // Allocated capacity
    unsigned int size_elem; // Size of each element in bytes
    char sorted;         // Flag indicating if the vector is sorted
} Vector;
```

---

## Initialization and Destruction

### `vector_init(Vector* vec, unsigned int size_elem)`

Initializes a vector for elements of size `size_elem`. Starts with capacity 1.

### `vector_destroy(Vector* vec)`

Frees memory used by the vector and resets its fields.

---

## Capacity Management

* `vector_resize(Vector* vec)` - Doubles vector capacity.
* `vector_reserve(Vector* vec, unsigned int additional)` - Reserves extra capacity.
* `vector_set_capacity(Vector* vec, unsigned int new_capacity)` - Sets vector capacity explicitly.
* `vector_shrink(Vector* vec)` - Shrinks capacity to current size.

**Accessors:**

```c
unsigned int vector_get_size(Vector* vec);
unsigned int vector_get_capacity(Vector* vec);
unsigned int vector_get_element_size(Vector* vec);
char vector_get_sorted(Vector* vec);
char vector_is_empty(Vector* vec);
```

---

## Element Access

* `vector_first(Vector* vec, void* el)` - Copies first element.
* `vector_last(Vector* vec, void* el)` - Copies last element.
* `vector_at(Vector* vec, unsigned int index, void* el)` - Copies element at index.

---

## Insertion

* `vector_push(Vector* vec, void* el)` - Adds element at end.
* `vector_push_front(Vector* vec, void* el)` - Adds element at start.
* `vector_push_at(Vector* vec, unsigned int index, void* el)` - Inserts element at index.

---

## Removal

* `vector_pop(Vector* vec, void* el)` - Removes last element.
* `vector_pop_front(Vector* vec, void* el)` - Removes first element.
* `vector_pop_at(Vector* vec, unsigned int index, void* el)` - Removes element at index.
* `vector_erase(Vector* vec, unsigned int index)` - Removes element without returning.

---

## Utility Functions

* `vector_clear(Vector* vec)` - Clears all elements; vector remains allocated.

---

## Example Usage

```c
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

    vector_destroy(&vec);
    return 0;
}
```

This module provides a flexible, standalone dynamic array implementation for efficient element management in C.
