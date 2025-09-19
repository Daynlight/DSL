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
#include <stdexcept>

// namespace Math
namespace Math{
class ModuloZ{
private:
  int number = 0;
  unsigned int modulo = 1;

private:
  constexpr void bound() noexcept;

public:
  constexpr ModuloZ();
  constexpr ModuloZ(unsigned int modulo);
  constexpr ModuloZ(unsigned int modulo, int number);
  
// conversion
  operator int() const;

// basic operations
  constexpr ModuloZ operator=(const ModuloZ &second) noexcept;
  constexpr ModuloZ operator=(const int &second) noexcept;
  constexpr ModuloZ operator+(const ModuloZ &second) noexcept;
  constexpr ModuloZ operator+(const int &second) noexcept;
  constexpr ModuloZ operator-(const ModuloZ &second) noexcept;
  constexpr ModuloZ operator-(const int &second) noexcept;
  constexpr ModuloZ operator*(const ModuloZ &second) noexcept;
  constexpr ModuloZ operator*(const int &second) noexcept;
  constexpr ModuloZ operator/(const ModuloZ &second);
  constexpr ModuloZ operator/(const int &second);
  constexpr ModuloZ operator%(const ModuloZ &second);
  constexpr ModuloZ operator%(const int &second);

// basic operations with =
  constexpr void operator+=(const ModuloZ &second) noexcept;
  constexpr void operator+=(const int &second) noexcept;
  constexpr void operator-=(const ModuloZ &second) noexcept;
  constexpr void operator-=(const int &second) noexcept;
  constexpr void operator*=(const ModuloZ &second) noexcept;
  constexpr void operator*=(const int &second) noexcept;
  constexpr void operator/=(const ModuloZ &second);
  constexpr void operator/=(const int &second);
  constexpr void operator%=(const ModuloZ &second);
  constexpr void operator%=(const int &second);
  
// logical operation
  constexpr bool operator==(const ModuloZ &second) const noexcept;
  constexpr bool operator==(const int &second) const noexcept;
  constexpr bool operator!=(const ModuloZ &second) const noexcept;
  constexpr bool operator!=(const int &second) const noexcept;
  constexpr bool operator>(const ModuloZ &second) const noexcept;
  constexpr bool operator>(const int &second) const noexcept;
  constexpr bool operator<(const ModuloZ &second) const noexcept;
  constexpr bool operator<(const int &second) const noexcept;
  constexpr bool operator>=(const ModuloZ &second) const noexcept;
  constexpr bool operator>=(const int &second) const noexcept;
  constexpr bool operator<=(const ModuloZ &second) const noexcept;
  constexpr bool operator<=(const int &second) const noexcept;
};
};

#include "ModuloZ.hpp"