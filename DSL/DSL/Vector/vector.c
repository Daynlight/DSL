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
  memset(vec->data, 0, vec->capacity * vec->size_elem);
  vec->size = 0;
  vec->sorted = 1;
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

void vector_foreach(Vector *vec, void (*func)(void *, void *), void *user_data) {
  if (!vec || !vec->data || !func) return;

  for (unsigned int i = 0; i < vec->size; i++) {
    void *elem = (char*)vec->data + i * vec->size_elem;
    func(elem, user_data);
  }
}

unsigned int vector_find(Vector *vec, void *el, int (*cmp)(const void *, const void *))
{
  if (!vec || !vec->data || !el) return 0;
  if(vec->sorted == 1) return vector_bfind(vec, el, cmp);
  for (unsigned int i = 0; i < vec->size; i++) {
    void *elem = (char*)vec->data + i * vec->size_elem;
    if (cmp(elem, el) == 0) return i;
  }
  return vec->size;
};

unsigned int vector_bfind(Vector *vec, void *el, int (*cmp)(const void *, const void *)){
  if (!vec || !vec->data || !el || !cmp || vec->size == 0) 
      return vec->size;

  unsigned int left = 0;
  unsigned int right = vec->size - 1;

  while (left <= right) {
    unsigned int mid = left + (right - left) / 2;
    void *mid_elem = (char*)vec->data + mid * vec->size_elem;
    int comparison = cmp(mid_elem, el);

    if (comparison == 0) {
      return mid; 
    } else if (comparison < 0) {
      left = mid + 1;
    } else {
      if (mid == 0) break;
      right = mid - 1;
    }
  }

  return vec->size;
};

unsigned int vector_find_if(Vector *vec, int (*predicate)(const void *)) {
  if (!vec || !vec->data || !predicate || vec->size_elem == 0) return vec->size;
  if(vec->sorted == 1) return vector_bfind_if(vec, predicate);
  for (unsigned int i = 0; i < vec->size; i++) {
    void *elem = (char*)vec->data + i * vec->size_elem;
    if (predicate(elem)) return i;
  }
  return vec->size;
};

unsigned int vector_bfind_if(Vector *vec, int (*predicate)(const void *)){
  if (!vec || !vec->data || !predicate || vec->size_elem == 0) return vec->size;

  unsigned int left = 0;
  unsigned int right = vec->size;
  unsigned int result = vec->size;

  while (left < right) {
    unsigned int mid = left + (right - left) / 2;
    void *mid_elem = (char*)vec->data + mid * vec->size_elem;

    if (predicate(mid_elem)) {
        result = mid;
        right = mid;
    } else {
        left = mid + 1;
    }
  }

  return result;
};

char vector_contains(Vector *vec, void *el, int (*cmp)(const void *, const void *)) {
  if (!vec || !vec->data || vec->size == 0) return 0;
  if(vec->sorted == 1) return vector_bcontains(vec, el, cmp);
  for (size_t i = 0; i < vec->size; i++) {
    void *elem = (char*)vec->data + i * vec->size_elem;
    if (cmp(elem, el) == 0) return 1;
  }
  return 0;
}

char vector_bcontains(Vector *vec, void *el, int (*cmp)(const void *, const void *)) {
  if (!vec || !vec->data || vec->size == 0) return 0;

  size_t left = 0;
  size_t right = vec->size - 1;

  while (left <= right) {
    size_t mid = left + (right - left) / 2;
    void *mid_elem = (char*)vec->data + mid * vec->size_elem;
    int comparison = cmp(mid_elem, el);

    if (comparison == 0) {
        return 1;
    } else if (comparison < 0) {
        left = mid + 1;
    } else {
        if (mid == 0) break;
        right = mid - 1;
    }
  }

  return 0;
};

void vector_swap(Vector *a, Vector *b) {
  if (!a || !b) return;

  Vector temp = *a;
  *a = *b;
  *b = temp; 
};

void vector_map(Vector *vec, void (*func)(void *)) {
  if (!vec || !vec->data || !func) return;

  for (unsigned int i = 0; i < vec->size; i++) {
    void *elem = (char*)vec->data + i * vec->size_elem;
    func(elem);
  }
};

void vector_qsort(Vector *vec, int (*cmp)(const void *, const void *)) {
  if (!vec || !vec->data || vec->size < 2) {
    vec->sorted = 1;
    return;
  };
  vector_qsort_helper(vec, cmp, 0, vec->size - 1);
  vec->sorted = 1;
};

void vector_qsort_helper(Vector *vec, int (*cmp)(const void *, const void *), int low, int high) {
  if (low < high) {
    int pi = vector_qsort_partition(vec, cmp, low, high);
    if (pi > 0) vector_qsort_helper(vec, cmp, low, pi - 1); 
    vector_qsort_helper(vec, cmp, pi + 1, high);
  }
};

int vector_qsort_partition(Vector *vec, int (*cmp)(const void *, const void *), int low, int high) {
  void *pivot = (char*)vec->data + high * vec->size_elem;
  int i = low;
  for (int j = low; j < high; j++) {
    void *elem = (char*)vec->data + j * vec->size_elem;
    if (cmp(elem, pivot) < 0) {
      char *a = (char*)vec->data + i * vec->size_elem;
      char *b = (char*)vec->data + j * vec->size_elem;
      char tmp[vec->size_elem];
      memcpy(tmp, a, vec->size_elem);
      memcpy(a, b, vec->size_elem);
      memcpy(b, tmp, vec->size_elem);
      i++;
    }
  }
  char *a = (char*)vec->data + i * vec->size_elem;
  char *b = (char*)vec->data + high * vec->size_elem;
  char tmp[vec->size_elem];
  memcpy(tmp, a, vec->size_elem);
  memcpy(a, b, vec->size_elem);
  memcpy(b, tmp, vec->size_elem);
  return i;
};
