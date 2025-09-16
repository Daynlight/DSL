#include "ModuloZ.h"

constexpr void Math::ModuloZ::bound() noexcept {
  number %= modulo;
  if (number < 0) number += modulo;
};

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
};

Math::ModuloZ::operator unsigned int() const {
  if(number < 0) return 0;
  return static_cast<unsigned int>(number);
};

constexpr Math::ModuloZ Math::ModuloZ::operator=(const ModuloZ &second) noexcept {
  number = second.number;
  modulo = second.modulo;
  return *this;
}

constexpr Math::ModuloZ Math::ModuloZ::operator=(const unsigned int &second) noexcept {
  this->number = second;
  return *this;
};

constexpr Math::ModuloZ Math::ModuloZ::operator+(const ModuloZ &second) noexcept {
  Math::ModuloZ eq(this->modulo);
  eq.number = this->number + second.number;
  eq.bound();
  return eq;
};

constexpr Math::ModuloZ Math::ModuloZ::operator+(const unsigned int &second) noexcept {
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

constexpr Math::ModuloZ Math::ModuloZ::operator-(const unsigned int &second) noexcept {
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

constexpr Math::ModuloZ Math::ModuloZ::operator*(const unsigned int &second) noexcept {
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

constexpr Math::ModuloZ Math::ModuloZ::operator/(const unsigned int &second) {
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

constexpr Math::ModuloZ Math::ModuloZ::operator%(const unsigned int &second) {
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

constexpr void Math::ModuloZ::operator+=(const unsigned int &second) noexcept {
  this->number = this->number + second;
  bound();
};

constexpr void Math::ModuloZ::operator-=(const ModuloZ &second) noexcept {
  this->number = this->number - second.number;
  bound();
};

constexpr void Math::ModuloZ::operator-=(const unsigned int &second) noexcept {
  this->number = this->number - second;
  bound();
};

constexpr void Math::ModuloZ::operator*=(const ModuloZ &second) noexcept {
  this->number = this->number * second.number;
  bound();
};

constexpr void Math::ModuloZ::operator*=(const unsigned int &second) noexcept {
  this->number = this->number * second;
  bound();
};

constexpr void Math::ModuloZ::operator/=(const ModuloZ &second) {
  if(!second.number)
    throw std::logic_error("Can't divide by 0");

  this->number = this->number / second.number;
  bound();
};

constexpr void Math::ModuloZ::operator/=(const unsigned int &second) {
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

constexpr void Math::ModuloZ::operator%=(const unsigned int &second) {
  if(!second)
    throw std::logic_error("Can't modulo by 0");

  this->number = this->number % second;
  bound();
};




constexpr bool Math::ModuloZ::operator==(const ModuloZ &second) const noexcept {
  return this->number == second.number;
};

constexpr bool Math::ModuloZ::operator==(const unsigned int &second) const noexcept {
  return this->number == second;
};

constexpr bool Math::ModuloZ::operator!=(const ModuloZ &second) const noexcept {
  return this->number != second.number;
};

constexpr bool Math::ModuloZ::operator!=(const unsigned int &second) const noexcept {
  return this->number != second;
};

constexpr bool Math::ModuloZ::operator>=(const ModuloZ &second) const noexcept {
  return this->number >= second.number;
};

constexpr bool Math::ModuloZ::operator>=(const unsigned int &second) const noexcept {
  return this->number >= second;
};

constexpr bool Math::ModuloZ::operator<=(const ModuloZ &second) const noexcept {
  return this->number <= second.number;
};

constexpr bool Math::ModuloZ::operator<=(const unsigned int &second) const noexcept {
  return this->number <= second;
};

constexpr bool Math::ModuloZ::operator>(const ModuloZ &second) const noexcept {
  return this->number > second.number;
};

constexpr bool Math::ModuloZ::operator>(const unsigned int &second) const noexcept {
  return this->number > second;
};

constexpr bool Math::ModuloZ::operator<(const ModuloZ &second) const noexcept {
  return this->number < second.number;
};

constexpr bool Math::ModuloZ::operator<(const unsigned int &second) const noexcept {
  return this->number < second;
};