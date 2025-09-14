/*
 * MyProgram - short description
 * Copyright (C) 2025 Your Name
 *
 * This file is part of MyProgram.
 *
 * MyProgram is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MyProgram is distributed in the hope that it will be us eful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


 

#include "vector.h"

///#################### memory #####################///
///[ | | | | | | | | | | | | | | | | | | | | | | | ]///
/// ^ first                                  last ^///

void vector_init(Vector* vec, unsigned int size_elem) {
  if(!vec) return;
  vec->capacity = 1;
  vec->buckets = 0;
  vec->size = 0;
  vec->size_elem = size_elem;
  vec->first = 0;
  vec->last = vec->capacity - 1;
  vec->sorted = 1;
  vec->hashed = 0;
  vec->data = calloc(vec->capacity, size_elem);
  vec->hashmap = NULL;
  if (!vec->data) vec->capacity = 0;
};

void vector_destroy(Vector *vec) {
  if(!vec) return;
  free(vec->data);
  free(vec->hashmap);
  memset(vec, 0, sizeof(Vector));
};

void vector_set_capacity(Vector *vec, unsigned int new_capacity) {
  if (!vec) return;
  if (vec->size > new_capacity) new_capacity = vec->size;

  void* newData = calloc(new_capacity, vec->size_elem);
  if(!newData) return;
  unsigned int first_size = vec->last + 1;
  unsigned int second_size = vec->first > 0 ? vec->capacity - vec->first : 0;

  // Copy First Part
  if(first_size)
    memcpy(newData, vec->data, first_size * vec->size_elem);
  
  // Copy Second Part
  if(second_size){
    void* src = (char*)vec->data + (vec->capacity - second_size) * vec->size_elem;
    void* dest = (char*)newData + (new_capacity - second_size) * vec->size_elem;
    memcpy(dest, src, second_size * vec->size_elem);
  }

  // Set Variables
  if(!newData) return;
  free(vec->data);
  vec->data = newData;
  vec->capacity = new_capacity;
  vec->first = vec->first > 0 ? new_capacity - second_size : 0;
  vec->hashed = 0;
};

void vector_resize(Vector *vec) {
  if (!vec) return;
  unsigned int new_capacity = vec->capacity ? vec->capacity * 2 : 1;
  vector_set_capacity(vec, new_capacity);
};

void vector_reserve(Vector *vec, unsigned int additional) {
  if (!vec || additional == 0) return;
  vector_set_capacity(vec, vec->capacity + additional);
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
  if (!vec->data) return;
  if(vec->size >= vec->capacity) vector_resize(vec);
  
  vec->last = vec->last >= vec->capacity - 1 ? 0 : vec->last + 1;
  memcpy((char*)vec->data + vec->last * vec->size_elem, el, vec->size_elem);

  vec->size++;
  vec->sorted = 0;
};

void vector_push_front(Vector *vec, void *el) {
  if (!vec || !el) return;
  if (!vec->data) return;
  if (vec->size >= vec->capacity) vector_resize(vec);

  vec->first = vec->first <= 0 ? vec->capacity - 1 : vec->first - 1;
  memcpy((char*)vec->data + vec->first * vec->size_elem, el, vec->size_elem);
  
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

  vector_last(vec, el);
  vec->last = vec->last <= 0 ? vec->capacity - 1 : vec->last - 1;
  
  vec->size--;
}

void vector_pop_front(Vector *vec, void *el){
  if (!vec || !el) return;
  if (vec->size <= 0) return;
  if (!vec->data) return;

  vector_first(vec, el);
  vec->first = vec->first >= vec->capacity - 1 ? 0 : vec->first + 1;
  
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
  void* target = (char*)vec->data + vec->first * vec->size_elem;
  memcpy(el, target, vec->size_elem);
};

void vector_last(Vector *vec, void *el) {
  if(!vec || !el || vec->size == 0) return;
  void* target = (char*)vec->data + vec->last * vec->size_elem;
  memcpy(el, target, vec->size_elem);
};

void vector_at(Vector *vec, unsigned int index, void *el) {
  if(!vec || !el || !vec->data) return;
  if(vec->first < index && vec->last > index) return;
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
