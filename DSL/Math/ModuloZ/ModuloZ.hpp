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




#include "ModuloZ.h"

constexpr void Math::ModuloZ::bound() noexcept {
  if (number < 0) number += modulo * (modulo/number - 1) * (-1);
  number %= modulo;
};

constexpr Math::ModuloZ::ModuloZ() :number(0), modulo(1) {};

constexpr Math::ModuloZ::ModuloZ(unsigned int modulo) {
  if(!modulo)
    throw std::logic_error("Can't modulo by 0");
  this->modulo = modulo;
};

constexpr Math::ModuloZ::ModuloZ(unsigned int modulo, int number) {
  if(!modulo)
    throw std::logic_error("Can't modulo by 0");
  this->modulo = modulo;
  this->number = number;
  bound();
};

Math::ModuloZ::operator int() const {
  if(number < 0) return 0;
  return static_cast<int>(number);
};

constexpr Math::ModuloZ Math::ModuloZ::operator=(const ModuloZ &second) noexcept {
  number = second.number;
  modulo = second.modulo;
  return *this;
}

constexpr Math::ModuloZ Math::ModuloZ::operator=(const int &second) noexcept {
  this->number = second;
  return *this;
};

constexpr Math::ModuloZ Math::ModuloZ::operator+(const ModuloZ &second) noexcept {
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number + second.number;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator+(const int &second) noexcept {
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number + second;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator-(const ModuloZ &second) noexcept {
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number - second.number;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator-(const int &second) noexcept {
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number - second;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator*(const ModuloZ &second) noexcept {
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number * second.number;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator*(const int &second) noexcept {
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number * second;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator/(const ModuloZ &second) {
  if(!second.number)
    throw std::logic_error("Can't divide by 0");

  Math::ModuloZ eq(this->modulo);
  eq.number = this->number / second.number;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator/(const int &second) {
  if(!second)
    throw std::logic_error("Can't divide by 0");

  Math::ModuloZ eq(this->modulo);
  eq.number = this->number / second;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator%(const ModuloZ &second) {
  if(!second.number)
    throw std::logic_error("Can't modulo by 0");
    
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number / second.number;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator%(const int &second) {
  if(!second)
    throw std::logic_error("Can't modulo by 0");
    
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number / second;
  eq.bound();
  return eq;
};




constexpr void Math::ModuloZ::operator+=(const ModuloZ &second) noexcept {
  this->number = this->number + second.number;
  bound();
};

constexpr void Math::ModuloZ::operator+=(const int &second) noexcept {
  this->number = this->number + second;
  bound();
};

constexpr void Math::ModuloZ::operator-=(const ModuloZ &second) noexcept {
  this->number = this->number - second.number;
  bound();
};

constexpr void Math::ModuloZ::operator-=(const int &second) noexcept {
  this->number = this->number - second;
  bound();
};

constexpr void Math::ModuloZ::operator*=(const ModuloZ &second) noexcept {
  this->number = this->number * second.number;
  bound();
};

constexpr void Math::ModuloZ::operator*=(const int &second) noexcept {
  this->number = this->number * second;
  bound();
};

constexpr void Math::ModuloZ::operator/=(const ModuloZ &second) {
  if(!second.number)
    throw std::logic_error("Can't divide by 0");

  this->number = this->number / second.number;
  bound();
};

constexpr void Math::ModuloZ::operator/=(const int &second) {
  if(!second)
    throw std::logic_error("Can't divide by 0");
    
  this->number = this->number / second;
  bound();
};

constexpr void Math::ModuloZ::operator%=(const ModuloZ &second) {
  if(!second.number)
    throw std::logic_error("Can't modulo by 0");

  this->number = this->number % second.number;
  bound();
};

constexpr void Math::ModuloZ::operator%=(const int &second) {
  if(!second)
    throw std::logic_error("Can't modulo by 0");

  this->number = this->number % second;
  bound();
};




constexpr bool Math::ModuloZ::operator==(const ModuloZ &second) const noexcept {
  return this->number == second.number;
};

constexpr bool Math::ModuloZ::operator==(const int &second) const noexcept {
  return this->number == second;
};

constexpr bool Math::ModuloZ::operator!=(const ModuloZ &second) const noexcept {
  return this->number != second.number;
};

constexpr bool Math::ModuloZ::operator!=(const int &second) const noexcept {
  return this->number != second;
};

constexpr bool Math::ModuloZ::operator>=(const ModuloZ &second) const noexcept {
  return this->number >= second.number;
};

constexpr bool Math::ModuloZ::operator>=(const int &second) const noexcept {
  return this->number >= second;
};

constexpr bool Math::ModuloZ::operator<=(const ModuloZ &second) const noexcept {
  return this->number <= second.number;
};

constexpr bool Math::ModuloZ::operator<=(const int &second) const noexcept {
  return this->number <= second;
};

constexpr bool Math::ModuloZ::operator>(const ModuloZ &second) const noexcept {
  return this->number > second.number;
};

constexpr bool Math::ModuloZ::operator>(const int &second) const noexcept {
  return this->number > second;
};

constexpr bool Math::ModuloZ::operator<(const ModuloZ &second) const noexcept {
  return this->number < second.number;
};

constexpr bool Math::ModuloZ::operator<(const int &second) const noexcept {
  return this->number < second;
};