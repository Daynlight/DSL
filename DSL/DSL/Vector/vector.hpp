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
 * MyProgram is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */




#include "vector.h"

template<typename T>
Vector<T>::Vector() {
  capacity = 1;
  buckets = 0;
  
  size = 0;

  head = 0;
  back = capacity - 1;
  
  sorted = 1;
  hashed = 0;
  
  data = new T[capacity];
  hashmap = NULL;
  if (!data) 
    throw std::runtime_error("Can't create data ptr in constructor");
};

template<typename T>
Vector<T>::~Vector() noexcept {
  delete data;
  delete hashmap;
};

template<typename T>
void Vector<T>::setCapacity(unsigned int new_capacity) {
  if (size > new_capacity) 
    new_capacity = size;

  T* newData = new T[new_capacity];
  if(!newData) 
    throw std::runtime_error("Can't create newData in setCapicity");

  unsigned int first_size = head + 1;
  unsigned int second_size = back > 0 ? capacity - back : 0;

  // Copy First Part
  if(first_size)
    for(unsigned int i = 0; i < first_size; i++)
      newData[i] = data[i];
    
  // Copy Second Part
  if(second_size)
    for(unsigned int i = capacity - second_size; i < new_capacity; i++)
      newData[i] = data[i];

  // Set Variables
  delete data;
  data = newData;
  capacity = new_capacity;
  back = back > 0 ? new_capacity - second_size : 0;
};

template <typename T>
void Vector<T>::reserve(unsigned int additional){
  if (additional == 0) 
    return;
  setCapacity(capacity + additional);
};

template<typename T>
void Vector<T>::resize() {
  unsigned int new_capacity = capacity ? capacity * 2 : 1;
  setCapacity(new_capacity);
};

template<typename T>
void Vector<T>::shrink() {
  setCapacity(size);
};

template<typename T>
constexpr unsigned int Vector<T>::getSize() const noexcept {
  return size;
};

template<typename T>
constexpr unsigned int Vector<T>::getCapacity() const noexcept {
  return capacity;
};

template<typename T>
constexpr bool Vector<T>::getSorted() const noexcept {
  return sorted;
};

template<typename T>
constexpr bool Vector<T>::isEmpty() const noexcept {
  return size == 0;
};

template<typename T>
void Vector<T>::pushHead(T el) {
  if(!data) 
    throw std::runtime_error("Can't pushHead, data is nullptr");

  if(size >= capacity) 
    resize();
  
  head = head >= capacity - 1 ? 0 : head + 1;
  data[head] = el;
  
  size++;
  sorted = 0;
};

template<typename T>
void Vector<T>::pushFront(T el) {
  if(!data) 
    throw std::runtime_error("Can't pushFront, data is nullptr");

  if(size >= capacity) 
    resize();

  back = back <= 0 ? capacity - 1 : back - 1;
  data[back] = el;

  size++;
  sorted = 0;
};

template<typename T>
void Vector<T>::pushAt(unsigned int index, T el) {
  if(!data)
    throw std::runtime_error("Can't pushAt, data is nullptr");
    
  if(index > head && index < back) 
    throw std::logic_error("Can't pushAt, index out of range");

  if(size >= capacity)
    resize();

  if(index < head){
    for(unsigned int i = index; i < head; i++)
      data[i + 1] = data[i];
    data[index] = el;
    head++;
  }
  else {
    for(unsigned int i = index; i > back; i--)
      data[i - 1] = data[i];
    data[index] = el;
    back--;
  }

  size++;
  sorted = 0;
}

template<typename T>
const T Vector<T>::popHead() {
  if(size <= 0) 
    throw std::out_of_range("Can't popHead, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't popHead, data is nullptr");

  T temp = last();
  head = head <= 0 ? capacity - 1 : head - 1;
  
  size--;
  return temp;
}

template<typename T>
const T Vector<T>::popBack(){
  if (size <= 0) 
    throw std::out_of_range("Can't popHead, vector is empty");
  
  if (!data) 
    throw std::runtime_error("Can't popBack, data is nullptr");

  T temp = first();
  back = back >= capacity - 1 ? 0 : back + 1;
  
  size--;
  return temp;
};

template<typename T>
const T Vector<T>::popAt(unsigned int index) {
  if(index > head && index < back) 
    throw std::out_of_range("Can't popAt, index out of range");
  
  if (!data) 
    throw std::runtime_error("Can't popAt, data is nullptr");

  T temp = data[index];

  if(index < head){
    for(unsigned int i = index; i < head; i++)
      data[i] = data[i + 1];
      head--;
  }
  else {
    for(unsigned int i = index; i > back; i--)
      data[i] = data[i - 1];
    back++;
  }

  size--;
  return temp;
};

template<typename T>
void Vector<T>::clear() noexcept {
  delete data;
  data = new T[capacity];
  size = 0;
  sorted = 1;
}

template<typename T>
void Vector<T>::erase(unsigned int index) noexcept {
  if(index > head && index < back) 
    return;

  if (!data) 
    return;

  popAt(index);
};

template<typename T>
T& Vector<T>::first() const {
  if(isEmpty()) 
    throw std::out_of_range("Can't first, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't first, data is nullptr");

  return data[back];
};

template<typename T>
T& Vector<T>::last() const {
  if(isEmpty()) 
    throw std::out_of_range("Can't last, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't last, data is nullptr");

  return data[head];
};

template<typename T>
T& Vector<T>::at(unsigned int index) const {
  if(index > head && index < back) 
    throw std::out_of_range("Can't at, index out of range");

  if (!data) 
    throw std::runtime_error("Can't at, data is nullptr");

  return data[index];
};

// void vector_foreach(, void (*func)(void *, void *), void *user_data) {
//   if (!vec || !vec->data || !func) return;

//   for (unsigned int i = 0; i < vec->size; i++) {
//     void *elem = (char*)vec->data + i * vec->size_elem;
//     func(elem, user_data);
//   }
// }

// unsigned int vector_find(, void *el, int (*cmp)(const void *, const void *))
// {
//   if (!vec || !vec->data || !el) return 0;
//   if(vec->sorted == 1) return vector_bfind(vec, el, cmp);
//   for (unsigned int i = 0; i < vec->size; i++) {
//     void *elem = (char*)vec->data + i * vec->size_elem;
//     if (cmp(elem, el) == 0) return i;
//   }
//   return vec->size;
// };

// unsigned int vector_bfind(, void *el, int (*cmp)(const void *, const void *)){
//   if (!vec || !vec->data || !el || !cmp || vec->size == 0) 
//       return vec->size;

//   unsigned int left = 0;
//   unsigned int right = vec->size - 1;

//   while (left <= right) {
//     unsigned int mid = left + (right - left) / 2;
//     void *mid_elem = (char*)vec->data + mid * vec->size_elem;
//     int comparison = cmp(mid_elem, el);

//     if (comparison == 0) {
//       return mid; 
//     } else if (comparison < 0) {
//       left = mid + 1;
//     } else {
//       if (mid == 0) break;
//       right = mid - 1;
//     }
//   }

//   return vec->size;
// };

// unsigned int vector_find_if(, int (*predicate)(const void *)) {
//   if (!vec || !vec->data || !predicate || vec->size_elem == 0) return vec->size;
//   if(vec->sorted == 1) return vector_bfind_if(vec, predicate);
//   for (unsigned int i = 0; i < vec->size; i++) {
//     void *elem = (char*)vec->data + i * vec->size_elem;
//     if (predicate(elem)) return i;
//   }
//   return vec->size;
// };

// unsigned int vector_bfind_if(, int (*predicate)(const void *)){
//   if (!vec || !vec->data || !predicate || vec->size_elem == 0) return vec->size;

//   unsigned int left = 0;
//   unsigned int right = vec->size;
//   unsigned int result = vec->size;

//   while (left < right) {
//     unsigned int mid = left + (right - left) / 2;
//     void *mid_elem = (char*)vec->data + mid * vec->size_elem;

//     if (predicate(mid_elem)) {
//         result = mid;
//         right = mid;
//     } else {
//         left = mid + 1;
//     }
//   }

//   return result;
// };

// char vector_contains(, void *el, int (*cmp)(const void *, const void *)) {
//   if (!vec || !vec->data || vec->size == 0) return 0;
//   if(vec->sorted == 1) return vector_bcontains(vec, el, cmp);
//   for (size_t i = 0; i < vec->size; i++) {
//     void *elem = (char*)vec->data + i * vec->size_elem;
//     if (cmp(elem, el) == 0) return 1;
//   }
//   return 0;
// }

// char vector_bcontains(, void *el, int (*cmp)(const void *, const void *)) {
//   if (!vec || !vec->data || vec->size == 0) return 0;

//   size_t left = 0;
//   size_t right = vec->size - 1;

//   while (left <= right) {
//     size_t mid = left + (right - left) / 2;
//     void *mid_elem = (char*)vec->data + mid * vec->size_elem;
//     int comparison = cmp(mid_elem, el);

//     if (comparison == 0) {
//         return 1;
//     } else if (comparison < 0) {
//         left = mid + 1;
//     } else {
//         if (mid == 0) break;
//         right = mid - 1;
//     }
//   }

//   return 0;
// };

// void vector_swap(Vector *a, Vector *b) {
//   if (!a || !b) return;

//   Vector temp = *a;
//   *a = *b;
//   *b = temp; 
// };

// void vector_map(, void (*func)(void *)) {
//   if (!vec || !vec->data || !func) return;

//   for (unsigned int i = 0; i < vec->size; i++) {
//     void *elem = (char*)vec->data + i * vec->size_elem;
//     func(elem);
//   }
// };

// void vector_qsort(, int (*cmp)(const void *, const void *)) {
//   if (!vec || !vec->data || vec->size < 2) {
//     vec->sorted = 1;
//     return;
//   };
//   vector_qsort_helper(vec, cmp, 0, vec->size - 1);
//   vec->sorted = 1;
// };

// void vector_qsort_helper(, int (*cmp)(const void *, const void *), int low, int high) {
//   if (low < high) {
//     int pi = vector_qsort_partition(vec, cmp, low, high);
//     if (pi > 0) vector_qsort_helper(vec, cmp, low, pi - 1); 
//     vector_qsort_helper(vec, cmp, pi + 1, high);
//   }
// };

// int vector_qsort_partition(, int (*cmp)(const void *, const void *), int low, int high) {
//   void *pivot = (char*)vec->data + high * vec->size_elem;
//   int i = low;
//   for (int j = low; j < high; j++) {
//     void *elem = (char*)vec->data + j * vec->size_elem;
//     if (cmp(elem, pivot) < 0) {
//       char *a = (char*)vec->data + i * vec->size_elem;
//       char *b = (char*)vec->data + j * vec->size_elem;
//       char tmp[vec->size_elem];
//       memcpy(tmp, a, vec->size_elem);
//       memcpy(a, b, vec->size_elem);
//       memcpy(b, tmp, vec->size_elem);
//       i++;
//     }
//   }
//   char *a = (char*)vec->data + i * vec->size_elem;
//   char *b = (char*)vec->data + high * vec->size_elem;
//   char tmp[vec->size_elem];
//   memcpy(tmp, a, vec->size_elem);
//   memcpy(a, b, vec->size_elem);
//   memcpy(b, tmp, vec->size_elem);
//   return i;
// };
