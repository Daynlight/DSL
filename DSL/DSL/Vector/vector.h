#ifndef PragmaVector
#define PragmaVector
#include <stdlib.h>
#include <string.h>

// Structure
typedef struct {
  void* data;
  unsigned int size;
  unsigned int capacity;
  unsigned int size_elem;
  char sorted;
} Vector;

// Constructor/Destructor
void vector_init(Vector* vec, unsigned int size_elem);
void vector_destroy(Vector* vec);

// Vector management
void vector_resize(Vector* vec);
void vector_reserve(Vector* vec, unsigned int additional);
void vector_set_capacity(Vector* vec, unsigned int new_capacity);
void vector_shrink(Vector* vec);

// Get Info
unsigned int vector_get_size(Vector* vec);
unsigned int vector_get_capacity(Vector* vec);
unsigned int vector_get_element_size(Vector* vec);
char vector_get_sorted(Vector* vec);
char vector_is_empty(Vector* vec);

// Pushers
void vector_push(Vector* vec, void* el);
void vector_push_front(Vector* vec, void* el);
void vector_push_at(Vector* vec, unsigned int index, void* el);

// Popers
void vector_pop(Vector* vec, void* el);
void vector_pop_front(Vector* vec, void* el);
void vector_pop_at(Vector* vec, unsigned int index, void* el);

// Removers
void vector_clear(Vector* vec);
void vector_erase(Vector* vec, unsigned int index);

// Get Without Removing
void vector_first(Vector* vec, void* el);
void vector_last(Vector* vec, void* el);
void vector_at(Vector* vec, unsigned int index, void* el);

// Algorithms
void vector_foreach(Vector* vec, void (*func)(void*, void*), void* user_data);
unsigned int vector_find(Vector* vec, void* el, int (*cmp)(const void*, const void*));
unsigned int vector_bfind(Vector* vec, void* el, int (*cmp)(const void*, const void*));
unsigned int vector_find_if(Vector* vec, int (*predicate)(const void*));
unsigned int vector_bfind_if(Vector* vec, int (*predicate)(const void*));
char vector_bcontains(Vector* vec, void* el, int (*cmp)(const void*, const void*));
void vector_swap(Vector* a, Vector* b);
void vector_map(Vector* vec, void (*func)(void*));
void vector_qsort(Vector* vec, int (*cmp)(const void*, const void*));
void vector_qsort_partition(Vector* vec, int (*cmp)(const void*, const void*));

#endif