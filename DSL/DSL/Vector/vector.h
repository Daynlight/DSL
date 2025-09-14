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



///#################### memory #####################///
///[ | | | | | | | | | | | | | | | | | | | | | | | ]///
/// ^ head                                   back ^ ///




#pragma once
#include <stdlib.h>
#include <stdexcept>
#include <string.h>

// HashMap for lookup
struct HashMap{
  unsigned int hash;
  unsigned int index;
};

// VectorClass
template<typename T>
class Vector {
private:
  T* data;
  Vector<HashMap>* hashmap = nullptr;
  unsigned int buckets;
  unsigned int capacity;
  unsigned int size;

  char sorted;
  char hashed;

  unsigned int head;
  unsigned int back;

public:
  // Constructor/Destructor
  Vector();    // O(1)
  ~Vector() noexcept;    // O(1)

  // Vector management
  void setCapacity(unsigned int new_capacity);    // O(log2(n))
  void reserve(unsigned int additional);    // O(log2(n))
  void resize();    // O(log2(n))
  void shrink();    // O(log2(n))

  // Get Info
  constexpr unsigned int getSize() const noexcept;    // O(1)
  constexpr unsigned int getCapacity() const noexcept;    // O(1)
  constexpr bool getSorted() const noexcept;    // O(1)
  constexpr bool isEmpty() const noexcept;    // O(1)

  // Pushers
  void pushHead(T el);    // O(log2(n))
  void pushFront(T el);    // O(log2(n))
  void pushAt(unsigned int index, T el);    // O(log2(n))

  // Popers
  const T popHead();    // O(1)
  const T popBack();    // O(1)
  const T popAt(unsigned int index);    // O(n)

  // Removers
  void clear() noexcept;    // O(1)
  void erase(unsigned int index) noexcept;    // O(n)

  // Get Without Removing       
  T& first() const;    // O(1)
  T& last() const;    // O(1)
  T& at(unsigned int index) const;    // O(1)

  // // Algorithms
  // void foreach(void (*func)(void*, void*), void* user_data);                      // O(n)
  // void map(void (*func)(void*));                                                  // O(n)
  // unsigned int find(void* el, int (*cmp)(const void*, const void*));              // O(n)
  // unsigned int find_if(int (*predicate)(const void*));                            // O(n)
  // char contains(void* el, int (*cmp)(const void*, const void*));                  // O(n)
  // unsigned int bfind(void* el, int (*cmp)(const void*, const void*));             // O(log n)
  // unsigned int bfind_if(int (*predicate)(const void*));                           // O(log n)
  // char bcontains(void* el, int (*cmp)(const void*, const void*));                 // O(log n)
  // void swap(Vector* a, Vector* b);                                                             // O(1)
  // void qsort(int (*cmp)(const void*, const void*));                               // O(n log n) worst O(n^2)
  // void qsort_helper(int (*cmp)(const void*, const void*), int low, int high);
  // int qsort_partition(int (*cmp)(const void*, const void*), int low, int high);
};

#include "vector.hpp"