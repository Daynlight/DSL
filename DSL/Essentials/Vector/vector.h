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



///#################### memory #####################///
///[ | | | | | | | | | | | | | | | | | | | | | | | ]///
/// ^ head                                   back ^ ///




#pragma once
#include <stdlib.h>
#include <stdexcept>
#include <string.h>
#include <vector>
#include <ostream>
#include "Math/ModuloZ/ModuloZ.h"

//namespace Essential
namespace Essentials {
// VectorClass
template<typename T>
class Vector {
private:
  T* data = nullptr;
  unsigned int capacity = 2;
  unsigned int size = 0;

protected:
  Math::ModuloZ _head = Math::ModuloZ(2, 1);
  Math::ModuloZ _back = Math::ModuloZ(2, 0);

private:
  constexpr bool inRange(unsigned int index) const noexcept;
  constexpr unsigned int toPhysicalIndex(int index) const noexcept;

public:
  // Constructor/Destructor
  Vector() noexcept;    // O(1)
  Vector(unsigned int capacity) noexcept;    // O(1)
  Vector(const Vector& other) noexcept;   // O(n)
  Vector(Vector&& other) noexcept;    // O(1)
  ~Vector() noexcept;    // O(1)

  // Vector management
  void setCapacity(unsigned int new_capacity);    // O(log2(n))
  void reorder();    // O(log2(n))
  void reserve(unsigned int additional);    // O(log2(n))
  void resize();    // O(log2(n))
  void shrink();    // O(log2(n))

  // Get Info
  constexpr unsigned int getSize() const noexcept;    // O(1)
  constexpr unsigned int getCapacity() const noexcept;    // O(1)
  constexpr bool isEmpty() const noexcept;    // O(1)
  // constexpr bool isFull() const noexcept;

  // Pushers
  void pushHead(T el);    // O(n)
  void pushBack(T el);    // O(n)
  void pushAt(int index, T el);    // O(n)

  // Popers
  const T popHead();    // O(1)
  const T popBack();    // O(1)
  const T popAt(int index);    // O(n)

  // Removers
  void clear() noexcept;    // O(1)
  void erase(int index) noexcept;    // O(n)

  // Get Without Removing       
  T& back();    // O(1)
  // const T& back() const;
  T& head();    // O(1)
  // const T& head() const;
  T& at(int index);    // O(1)
  // const T& at(int index) const;

  // operators
  // T& operator[](int index);
  // Vector& operator=(const Vector &second);
  // Vector& operator=(Vector &&second);
  // operator std::vector<T>(); 
  // friend std::ostream& operator<<(std::ostream& os, const Vector<T>& v);
  
  // compare
  // bool operator==(const Vector &second);
  // bool operator!=(const Vector &second);
  // bool operator>(const Vector &second);
  // bool operator>=(const Vector &second);
  // bool operator<(const Vector &second);
  // bool operator<=(const Vector &second);

  // save/load
  // const std::string& serialize();
  // void deserialize();
};
};

#include "vector.hpp"