#include "vector.h"

void vector_init(Vector* vec, unsigned int size_elem) {
  if(!vec) return;
  vec->capacity = 1;
  vec->size = 0;
  vec->size_elem = size_elem;
  vec->sorted = 1;
  vec->data = calloc(vec->capacity, size_elem);
  if (!vec->data) vec->capacity = 0;
};

void vector_destroy(Vector *vec) {
  if(!vec) return;
  free(vec->data);
  memset(vec, 0, sizeof(Vector));
};

void vector_resize(Vector *vec) {
  if (!vec) return;
  
  unsigned int new_capacity = vec->capacity ? vec->capacity * 2 : 1;
  void* temp = realloc(vec->data, new_capacity * vec->size_elem);
  if(!temp) return;
  
  vec->data = temp;
  vec->capacity = new_capacity;
};

void vector_reserve(Vector *vec, unsigned int additional) {
  if (!vec || additional == 0) return;
  
  unsigned int target_capacity = vec->capacity + additional;
  void* temp = realloc(vec->data, target_capacity * vec->size_elem);
  if(!temp) return;
  
  vec->data = temp;
  vec->capacity = target_capacity;
};

void vector_set_capacity(Vector *vec, unsigned int new_capacity) {
  if (!vec) return;
  
  if (vec->size > new_capacity) vec->size = new_capacity;

  if (new_capacity <= 0) new_capacity = 1;

  void* temp = realloc(vec->data, new_capacity * vec->size_elem);
  if(!temp) return;
  
  vec->data = temp;
  vec->capacity = new_capacity;
};

void vector_shrink(Vector *vec) {
  vector_set_capacity(vec, vec->size);
};

unsigned int vector_get_size(Vector *vec) {
  return vec->size;
};

unsigned int vector_get_capacity(Vector *vec) {
  return vec->capacity;
};

unsigned int vector_get_element_size(Vector *vec){
  return vec->size_elem;
};

char vector_get_sorted(Vector *vec){
  return vec->sorted;
};

char vector_is_empty(Vector *vec) {
  return vec->size == 0;
};

void vector_push(Vector *vec, void *el) {
  if(!vec || !el || !vec->data) return;

  if(vec->size >= vec->capacity) vector_resize(vec);
  if (!vec->data) return;

  void* target = (char*)vec->data + vec->size * vec->size_elem;
  memcpy(target, el, vec->size_elem);
  vec->size++;
  vec->sorted = 0;
};

void vector_push_front(Vector *vec, void *el) {
  if (!vec || !el) return;

  if (vec->size >= vec->capacity) vector_resize(vec);
  if (!vec->data) return;

  if (vec->size > 0) {
    memmove((char*)vec->data + vec->size_elem, vec->data, vec->size * vec->size_elem);
  }

  memcpy(vec->data, el, vec->size_elem);
  vec->size++;
  vec->sorted = 0;
};

void vector_push_at(Vector *vec, unsigned int index, void *el) {
    if (!vec || !el) return;
    if (index > vec->size) return;

    if (vec->size >= vec->capacity) vector_resize(vec);
    if (!vec->data) return;

    if (index < vec->size) {
        memmove(
            (char*)vec->data + (index + 1) * vec->size_elem,
            (char*)vec->data + index * vec->size_elem,
            (vec->size - index) * vec->size_elem
        );
    }

    memcpy((char*)vec->data + index * vec->size_elem, el, vec->size_elem);
    vec->size++;
    vec->sorted = 0;
}

void vector_pop(Vector *vec, void *el) {
  if(!vec || !el) return;
  if (!vec->data) return;

  if(vec->size <= 0) return;

  vec->size--;
  void* target = (char*)vec->data + vec->size * vec->size_elem;
  memcpy(el, target, vec->size_elem);
}

void vector_pop_front(Vector *vec, void *el){
  if (!vec || !el) return;
  if (vec->size <= 0) return;
  if (!vec->data) return;

  memcpy(el, vec->data, vec->size_elem);
    if (vec->size > 1)
      memmove(vec->data, (char*)vec->data + vec->size_elem, (vec->size - 1) * vec->size_elem);

  vec->size--;
};

void vector_pop_at(Vector *vec, unsigned int index, void *el) {
  if (!vec || !el) return;
  if (vec->size == 0 || index >= vec->size) return;
  if (!vec->data) return;

  memcpy(el, (char*)vec->data + index * vec->size_elem, vec->size_elem);

  if (index < vec->size - 1) {
    memmove((char*)vec->data + index * vec->size_elem,
            (char*)vec->data + (index + 1) * vec->size_elem,
            (vec->size - index - 1) * vec->size_elem);
  }

  vec->size--;
};

void vector_clear(Vector *vec) {
  if (!vec) return;
  vec->size = 0;
  memset(vec->data, 0, vec->capacity * vec->size_elem);
  vec->sorted;
}

void vector_erase(Vector *vec, unsigned int index) {
  if (!vec) return;
  if (vec->size == 0 || index >= vec->size) return;
  if (!vec->data) return;

  if (index < vec->size - 1) {
    memmove((char*)vec->data + index * vec->size_elem,
            (char*)vec->data + (index + 1) * vec->size_elem,
            (vec->size - index - 1) * vec->size_elem);
  }

  vec->size--;
};

void vector_first(Vector *vec, void *el) {
  if(!vec || !el || vec->size == 0) return;
  memcpy(el, vec->data, vec->size_elem);
};

void vector_last(Vector *vec, void *el) {
  if(!vec || !el || vec->size == 0) return;
  memcpy(el, (char*)vec->data + (vec->size - 1) * vec->size_elem, vec->size_elem);
};

void vector_at(Vector *vec, unsigned int index, void *el) {
  if(!vec || !el) return;
  if (!vec->data) return;

  if(vec->size <= index) return;
  
  void* target = (char*)vec->data + index * vec->size_elem;
  memcpy(el, target, vec->size_elem);
};
