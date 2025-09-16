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

template <typename T>
Essentials::Vector<T>::~Vector() noexcept {
  if(data) 
    delete[] data;
  if(hashmap) 
    delete hashmap;
};

template<typename T>
void Essentials::Vector<T>::setCapacity(unsigned int new_capacity) {
  if (size > new_capacity) 
    new_capacity = size;
  
  if(new_capacity < 2) new_capacity = 2;

  T* newData = new T[new_capacity];

  if(size){
    unsigned int first_chunk = std::min(size, capacity - back);
    unsigned int second_chunk = size - first_chunk;

    std::copy(data + back, data + back + first_chunk, newData);
    if (second_chunk)
        std::copy(data, data + second_chunk, newData + first_chunk);
  };

  // Set Variables
  delete[] data;
  data = newData;
  capacity = new_capacity;
  back = Math::ModuloZ(new_capacity, 0);
  head = size <= 0 ? Math::ModuloZ(new_capacity, new_capacity) 
                   : Math::ModuloZ(new_capacity, size);
  head -= 1;
};

template <typename T>
void Essentials::Vector<T>::reserve(unsigned int additional){
  if (additional == 0) 
    return;
  setCapacity(capacity + additional);
};

template<typename T>
void Essentials::Vector<T>::resize() {
  unsigned int new_capacity = capacity >= 2 ? capacity * 2 : 2;
  setCapacity(new_capacity);
};

template<typename T>
void Essentials::Vector<T>::shrink() {
  setCapacity(size);
};

template<typename T>
constexpr unsigned int Essentials::Vector<T>::getSize() const noexcept {
  return size;
};

template<typename T>
constexpr unsigned int Essentials::Vector<T>::getCapacity() const noexcept {
  return capacity;
};

template<typename T>
constexpr bool Essentials::Vector<T>::getSorted() const noexcept {
  return sorted;
};

template<typename T>
constexpr bool Essentials::Vector<T>::isEmpty() const noexcept {
  return size == 0;
};

template<typename T>
void Essentials::Vector<T>::pushHead(T el) {
  if(!data) 
    throw std::runtime_error("Can't pushHead, data is nullptr");

  if(size >= capacity) 
    resize();
  
  head += 1;
  data[head] = el;
  
  size++;
  sorted = 0;
};

template<typename T>
void Essentials::Vector<T>::pushBack(T el) {
  if(!data) 
    throw std::runtime_error("Can't pushBack, data is nullptr");

  if(size >= capacity) 
    resize();

  back -= 1;
  data[back] = el;

  size++;
  sorted = 0;
};

template <typename T>
constexpr inline bool Essentials::Vector<T>::inRange(unsigned int index) const noexcept {
  return back <= head ? (index >= back && index <= head)
                      : (index >= back || index <= head);
}

template<typename T>
void Essentials::Vector<T>::pushAt(unsigned int index, T el) {
  if(!inRange(index)) 
    throw std::logic_error("Can't pushAt, index out of range");
  
  if(!data)
    throw std::runtime_error("Can't pushAt, data is nullptr");

  if(size >= capacity)
    resize();

  if(index < head){
    Math::ModuloZ iterator = Math::ModuloZ(capacity, index);
    Math::ModuloZ iterator_next = Math::ModuloZ(capacity, index + 1);
    for(unsigned int i = index; i < head; i++){
      data[iterator] = data[iterator_next];
      iterator += 1;
      iterator_next += 1;
    }
    data[index] = el;
    head += 1;
  }
  else {
    Math::ModuloZ iterator = Math::ModuloZ(capacity, index);
    Math::ModuloZ iterator_next = Math::ModuloZ(capacity, index);
    iterator_next -= 1;
    for(unsigned int i = index; i > back; i--){
      data[iterator] = data[iterator_next];
      iterator -= 1;
      iterator_next -= 1;
    }
    back -= 1;
  }

  size++;
  sorted = 0;
}

template<typename T>
const T Essentials::Vector<T>::popHead() {
  if(size <= 0) 
    throw std::out_of_range("Can't popHead, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't popHead, data is nullptr");

  T temp = last();
  head -= 1;
  
  size--;
  return temp;
}

template<typename T>
const T Essentials::Vector<T>::popBack(){
  if (size <= 0) 
    throw std::out_of_range("Can't popBack, vector is empty");
  
  if (!data) 
    throw std::runtime_error("Can't popBack, data is nullptr");

  T temp = first();
  back += 1;
  
  size--;
  return temp;
};

template<typename T>
const T Essentials::Vector<T>::popAt(unsigned int index) {
  if(index > head && index < back) 
    throw std::out_of_range("Can't popAt, index out of range");
  
  if (!data) 
    throw std::runtime_error("Can't popAt, data is nullptr");

  T temp = data[index];

  if(index < head){
    Math::ModuloZ iterator = Math::ModuloZ(capacity, index);
    Math::ModuloZ iterator_next = Math::ModuloZ(capacity, index + 1);
    for(unsigned int i = index; i < head; i++){
      data[iterator] = data[iterator_next];
      iterator += 1;
      iterator_next += 1;
    } 
    head -= 1;
  }
  else {
    Math::ModuloZ iterator = Math::ModuloZ(capacity, index);
    Math::ModuloZ iterator_next = Math::ModuloZ(capacity, index);
    iterator_next -= 1;
    for(unsigned int i = index; i > back; i--){
      data[iterator] = data[iterator_next];
      iterator -= 1;
      iterator_next -= 1;
    }
    back += 1;
  }

  size--;
  return temp;
};

template<typename T>
void Essentials::Vector<T>::clear() noexcept {
  delete[] data;
  data = new T[capacity];
  size = 0;
  head = capacity - 1;
  back = 0;
  sorted = 1;
}

template<typename T>
void Essentials::Vector<T>::erase(unsigned int index) noexcept {
  if(index > head && index < back) 
    return;

  if (!data) 
    return;

  popAt(index);
};

template<typename T>
T& Essentials::Vector<T>::first() const {
  if(isEmpty()) 
    throw std::out_of_range("Can't first, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't first, data is nullptr");

  return data[back];
};

template<typename T>
T& Essentials::Vector<T>::last() const {
  if(isEmpty()) 
    throw std::out_of_range("Can't last, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't last, data is nullptr");

  return data[head];
};

template<typename T>
T& Essentials::Vector<T>::at(unsigned int index) const {
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
