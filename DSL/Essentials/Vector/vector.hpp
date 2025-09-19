/*
 * DSL - short description
 * Copyright (C) 2025 Your Name
 *
 * This file is part of DSL.
 *
 * DSL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */




#include "vector.h"

template<typename T>
Essentials::Vector<T>::Vector() noexcept 
  :data(new T[capacity]) {};

template <typename T>
inline Essentials::Vector<T>::Vector(unsigned int capacity) noexcept {
  setCapacity(capacity);
};

template <typename T>
inline Essentials::Vector<T>::Vector(const Vector &other) noexcept
  :capacity(other.capacity), size(other.size), 
  _head(other._head), _back(other._back), data(new T[other.capacity]) {
  if(size) {
    unsigned int first_chunk = std::min(size, capacity - _back);
    unsigned int second_chunk = size - first_chunk;

    std::copy(other.data + _back, other.data + _back + first_chunk, data);
    if (second_chunk)
        std::copy(other.data, other.data + second_chunk, data + first_chunk);
  };
};

template <typename T>
inline Essentials::Vector<T>::Vector(Vector &&other) noexcept 
  :capacity(other.capacity), size(other.size), 
  _head(other._head), _back(other._back), data(other.data){
  other.data = new T[2];
  other._head = Math::ModuloZ(2, 1);
  other._back = Math::ModuloZ(2, 0);
  other.size = 0;
  other.capacity = 2;
};

template<typename T>
Essentials::Vector<T>::~Vector() noexcept {
  delete[] data;
};

template<typename T>
void Essentials::Vector<T>::setCapacity(unsigned int new_capacity) {
  if (size > new_capacity) 
    new_capacity = size;
  
  if(new_capacity < 2) new_capacity = 2;

  T* newData = new T[new_capacity];

  if(size){
    unsigned int first_chunk = std::min(size, capacity - _back);
    unsigned int second_chunk = size - first_chunk;

    std::copy(data + _back, data + _back + first_chunk, newData);
    if (second_chunk)
        std::copy(data, data + second_chunk, newData + first_chunk);
  };

  // Set Variables
  delete[] data;
  data = newData;
  capacity = new_capacity;
  _back = Math::ModuloZ(new_capacity, 0);
  _head = size <= 0 ? Math::ModuloZ(new_capacity, new_capacity) 
                   : Math::ModuloZ(new_capacity, size);
  _head -= 1;
};

template <typename T>
inline void Essentials::Vector<T>::reorder() {
  setCapacity(capacity);
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
constexpr bool Essentials::Vector<T>::isEmpty() const noexcept {
  return size == 0;
};

template<typename T>
void Essentials::Vector<T>::pushHead(T el) {
  if(!data) 
    throw std::runtime_error("Can't pushHead, data is nullptr");

  if(size >= capacity) 
    resize();
  
  _head += 1;
  data[_head] = el;
  
  size++;
};

template<typename T>
void Essentials::Vector<T>::pushBack(T el) {
  if(!data) 
    throw std::runtime_error("Can't pushBack, data is nullptr");

  if(size >= capacity) 
    resize();

  _back -= 1;
  data[_back] = el;

  size++;
};

template <typename T>
constexpr inline bool Essentials::Vector<T>::inRange(unsigned int index) const noexcept {
  return _back <= _head ? (index >= _back && index <= _head)
                        : (index >= _back || index <= _head);
}

template <typename T>
inline constexpr unsigned int Essentials::Vector<T>::toPhysicalIndex(int index) const noexcept {
  unsigned int p_index;
  if(index >= 0)
    p_index = Math::ModuloZ(capacity, _back + index);
  else
    p_index = Math::ModuloZ(capacity, _head + index + 1);
  return p_index;
}

template<typename T>
void Essentials::Vector<T>::pushAt(int index, T el) {
  unsigned int p_index = toPhysicalIndex(index);

  if(!inRange(index)) 
    throw std::logic_error("Can't pushAt, index out of range");
  
  if(!data)
    throw std::runtime_error("Can't pushAt, data is nullptr");

  if(size >= capacity)
    resize();

  Math::ModuloZ iterator = Math::ModuloZ(capacity, _head);
  Math::ModuloZ iterator_next = Math::ModuloZ(capacity, _head + 1);
  Math::ModuloZ copy_size = _head - p_index + 1;
  for(unsigned int i = 0; i < copy_size; i++){
    data[iterator_next] = data[iterator];
    iterator -= 1;
    iterator_next -= 1;
  }
  data[p_index] = el;
  _head += 1;

  size++;
}

template<typename T>
const T Essentials::Vector<T>::popHead() {
  if(size <= 0) 
    throw std::out_of_range("Can't popHead, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't popHead, data is nullptr");

  T temp = head();
  _head -= 1;
  
  size--;
  return temp;
}

template<typename T>
const T Essentials::Vector<T>::popBack(){
  if (size <= 0) 
    throw std::out_of_range("Can't popBack, vector is empty");
  
  if (!data) 
    throw std::runtime_error("Can't popBack, data is nullptr");

  T temp = back();
  _back += 1;
  
  size--;
  return temp;
};

template<typename T>
const T Essentials::Vector<T>::popAt(int index) {
  unsigned int p_index = toPhysicalIndex(index);

  if(!inRange(index)) 
    throw std::out_of_range("Can't popAt, index out of range");
  
  if (!data) 
    throw std::runtime_error("Can't popAt, data is nullptr");

  T temp = data[p_index];

  Math::ModuloZ iterator = Math::ModuloZ(capacity, p_index);
  Math::ModuloZ iterator_next = Math::ModuloZ(capacity, p_index + 1);
  Math::ModuloZ copy_size = _head - p_index + 1;
  for(unsigned int i = 0; i <= copy_size; i++){
    data[iterator] = data[iterator_next];
    iterator += 1;
    iterator_next += 1;
  } 
  _head -= 1;

  size--;
  return temp;
};

template<typename T>
void Essentials::Vector<T>::clear() noexcept {
  delete[] data;
  data = new T[capacity];
  size = 0;
  _head = capacity - 1;
  _back = 0;
}

template<typename T>
void Essentials::Vector<T>::erase(int index) noexcept {
  unsigned int p_index = toPhysicalIndex(index);

  if(!inRange(p_index)) 
    return;

  if (!data) 
    return;

  popAt(index);
};

template<typename T>
T& Essentials::Vector<T>::back() {
  if(isEmpty()) 
    throw std::out_of_range("Can't first, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't first, data is nullptr");

  return data[_back];
};

template<typename T>
T& Essentials::Vector<T>::head() {
  if(isEmpty()) 
    throw std::out_of_range("Can't last, vector is empty");

  if (!data) 
    throw std::runtime_error("Can't last, data is nullptr");

  return data[_head];
};

template<typename T>
T& Essentials::Vector<T>::at(int index) {
  unsigned int p_index = toPhysicalIndex(index);
  
  if(!inRange(p_index)) 
    throw std::out_of_range("Can't at, index out of range");

  if (!data) 
    throw std::runtime_error("Can't at, data is nullptr");

  return data[p_index];
};
