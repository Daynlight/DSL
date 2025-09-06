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
void vector_init(Vector* vec, unsigned int size_elem);                                              // O(1)
void vector_destroy(Vector* vec);                                                                   // O(1)

// Vector management
void vector_resize(Vector* vec);                                                                    // O(n)
void vector_reserve(Vector* vec, unsigned int additional);                                          // O(n)
void vector_set_capacity(Vector* vec, unsigned int new_capacity);                                   // O(n)
void vector_shrink(Vector* vec);                                                                    // O(n)

// Get Info
unsigned int vector_get_size(Vector* vec);                                                          // O(1)
unsigned int vector_get_capacity(Vector* vec);                                                      // O(1)
unsigned int vector_get_element_size(Vector* vec);                                                  // O(1)
char vector_get_sorted(Vector* vec);                                                                // O(1)
char vector_is_empty(Vector* vec);                                                                  // O(1)

// Pushers
void vector_push(Vector* vec, void* el);                                                            // O(n)
void vector_push_front(Vector* vec, void* el);                                                      // O(n)
void vector_push_at(Vector* vec, unsigned int index, void* el);                                     // O(n)

// Popers
void vector_pop(Vector* vec, void* el);                                                             // O(1)
void vector_pop_front(Vector* vec, void* el);                                                       // O(n)
void vector_pop_at(Vector* vec, unsigned int index, void* el);                                      // O(n)

// Removers
void vector_clear(Vector* vec);                                                                     // O(n)
void vector_erase(Vector* vec, unsigned int index);                                                 // O(n)

// Get Without Removing       
void vector_first(Vector* vec, void* el);                                                           // O(1)
void vector_last(Vector* vec, void* el);                                                            // O(1)
void vector_at(Vector* vec, unsigned int index, void* el);                                          // O(1)

// Algorithms
void vector_foreach(Vector* vec, void (*func)(void*, void*), void* user_data);                      // O(n)
void vector_map(Vector* vec, void (*func)(void*));                                                  // O(n)
unsigned int vector_find(Vector* vec, void* el, int (*cmp)(const void*, const void*));              // O(n)
unsigned int vector_find_if(Vector* vec, int (*predicate)(const void*));                            // O(n)
char vector_contains(Vector* vec, void* el, int (*cmp)(const void*, const void*));                  // O(n)
unsigned int vector_bfind(Vector* vec, void* el, int (*cmp)(const void*, const void*));             // O(log n)
unsigned int vector_bfind_if(Vector* vec, int (*predicate)(const void*));                           // O(log n)
char vector_bcontains(Vector* vec, void* el, int (*cmp)(const void*, const void*));                 // O(log n)
void vector_swap(Vector* a, Vector* b);                                                             // O(1)
void vector_qsort(Vector* vec, int (*cmp)(const void*, const void*));                               // O(n log n) worst O(n^2)
void vector_qsort_helper(Vector *vec, int (*cmp)(const void*, const void*), int low, int high);
int vector_qsort_partition(Vector* vec, int (*cmp)(const void*, const void*), int low, int high);

#endif