#pragma once
#include "Tests/conf.h"
#include <fmt/core.h>
#include <fmt/color.h>
#include <string>
#include "dsl.h"

namespace UnitTests {
// =============== Helping Struct ===============
struct Point { int x, y; };
template<typename T>
class TestVector : public Essentials::Vector<T> {
public:
    unsigned int getBackIndex() { return this->_back; }
};


// =============== Comparators & Predicates ===============
int cmp_int(const int &a, const int &b) { return a - b; }
int is_even(const int &val) { return (val % 2) == 0; }
int point_cmp(const Point &a, const Point &b) { return a.x - b.x; }
void add(const int &el, int &ud) { ud += el; }
void multiply(int &el) { el *= 2; }


// ========================= ASSERT =========================
bool ASSERT_EQ_INT(std::string name, int expected, int actual) noexcept {
#ifdef LoudTests
  if(expected == actual) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={:d} got={:d}\n", name, expected, actual);  
#endif
  if(expected != actual)
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={:d} got={:d}\n", name, expected, actual); 
  return expected == actual;
};

bool ASSERT_EQ_SIZE(std::string name, size_t expected, size_t actual) noexcept {
#ifdef LoudTests
  if(expected == actual) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={} got={}\n", name, expected, actual);  
#endif
  if(expected != actual)
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={} got={}\n", name, expected, actual);
  return expected == actual;
};

bool ASSERT_EQ_FLOAT(std::string name, float expected, float actual, float eps) noexcept {
  float diff = std::abs(expected - actual);
#ifdef LoudTests
  if(diff < eps) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s} exp={} got={} (eps={})\n", name, expected, actual, eps); 
#endif   
  if(diff >= eps)
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s} exp={} got={} (eps={})\n", name, expected, actual, eps);
  return diff < eps;
};

bool ASSERT_TRUE(std::string name, bool cond) noexcept {
#ifdef LoudTests
  if(cond) 
    fmt::print(fg(fmt::color::green), "[PASS] {:s}\n", name);
#endif
  if(!cond)     
    fmt::print(fg(fmt::color::red),   "[FAIL] {:s}\n", name);
  return cond;
};






//=========================================================
//========================= Tests =========================
//=========================================================

// ===== INIT & DESTROY =====
void test_vector_init(unsigned int* total, unsigned int* passed) {
  *total += 12;
  Essentials::Vector<int> v;
  Essentials::Vector<int> v2(20);
  v2.pushHead(2);
  v2.pushHead(3);
  Essentials::Vector<int> v3(v2);
  
  if(ASSERT_EQ_SIZE("default constructor, size", 0, v.getSize()))*passed += 1;
  if(ASSERT_TRUE("default constructor, capacity>=1", v.getCapacity() >= 1))*passed += 1;
  
  if(ASSERT_EQ_SIZE("constructor with capacity, size", 2, v2.getSize()))*passed += 1;
  if(ASSERT_TRUE("constructor with capacity, capacity>=20", v2.getCapacity() >= 20))*passed += 1;
  
  if(ASSERT_EQ_SIZE("copy constructor, copy size", 2, v2.getSize()))*passed += 1;
  if(ASSERT_TRUE("copy constructor, capacity>=20", v2.getCapacity() >= 20))*passed += 1;
  if(ASSERT_EQ_INT("copy constructor, at 0", 2, v3.at(0)))*passed += 1;
  if(ASSERT_EQ_INT("copy constructor, at 1", 3, v3.at(1)))*passed += 1;
  v3.pushHead(4);
  if(ASSERT_EQ_SIZE("copy constructor edit copy, original size", 2, v2.getSize()))*passed += 1;
  if(ASSERT_EQ_SIZE("copy constructor edit copy, copy size", 3, v3.getSize()))*passed += 1;

  Essentials::Vector<int> v4(std::move(v2));
  if(ASSERT_EQ_SIZE("move constructor, original size", 0, v2.getSize()))*passed += 1;
  if(ASSERT_EQ_SIZE("move constructor, moved size", 2, v4.getSize()))*passed += 1;
}

// ===== RESIZE & CAPACITY =====
void test_vector_resize(unsigned int* total, unsigned int* passed) {
  *total += 2;
  Essentials::Vector<int> v;
  unsigned initial = v.getCapacity();
  v.resize();
  if(ASSERT_TRUE("resize doubles capacity", v.getCapacity() >= initial*2))*passed += 1;
  v.resize();
  if(ASSERT_TRUE("resize quadruple capacity", v.getCapacity() >= initial*4))*passed += 1;
}

void test_vector_reorder(unsigned int* total, unsigned int* passed) {
  *total += 2;
  TestVector<int> v;
  v.reserve(5);
  v.pushHead(21);
  v.pushHead(24);
  v.pushBack(21);
  unsigned int back = v.getBackIndex();
  if(ASSERT_TRUE("reorder check back before reorder", back != 0))*passed += 1;
  v.reorder();
  back = v.getBackIndex();
  if(ASSERT_TRUE("reorder check back after reorder", back == 0))*passed += 1;
}

void test_vector_reserve(unsigned int* total, unsigned int* passed) {
  *total += 1;
  Essentials::Vector<int> v;
  unsigned old = v.getCapacity();
  v.reserve(10);
  if(ASSERT_TRUE("reserve increases capacity", v.getCapacity()>=old+10))*passed += 1;
}

void test_vector_set_capacity(unsigned int* total, unsigned int* passed) {
  *total += 2;
  Essentials::Vector<int> v;
  v.setCapacity(20);
  if(ASSERT_EQ_SIZE("set_capacity=20", 20, v.getCapacity()))*passed += 1;
  v.setCapacity(2);
  if(ASSERT_EQ_SIZE("set_capacity=2", 2, v.getCapacity()))*passed += 1;
}

void test_vector_shrink(unsigned int* total, unsigned int* passed) {
  *total += 2;
  Essentials::Vector<int> v;
  v.setCapacity(60);
  v.pushHead(1);
  v.pushHead(1);
  v.pushHead(1);
  if(ASSERT_EQ_SIZE("shrink check before shrink", 60, v.getCapacity()))*passed += 1;
  v.shrink();
  if(ASSERT_EQ_SIZE("shrink size==capacity", v.getSize(), v.getCapacity()))*passed += 1;
}

// ===== PUSH METHODS =====
void test_vector_push_head(unsigned int* total, unsigned int* passed) {
  *total += 1;
  Essentials::Vector<int> v;
  v.reserve(10);
  v.pushHead(12);
  v.pushHead(32);
  v.pushHead(54);
  v.pushHead(11);
  v.pushHead(77);
  if(ASSERT_EQ_INT("push head value", 77, v.head()))*passed += 1;
}

void test_vector_push_back(unsigned int* total, unsigned int* passed) {
  *total += 1;
  Essentials::Vector<int> v;
  v.pushBack(1);
  v.pushBack(2);
  v.pushBack(67);
  v.pushBack(99);
  v.pushBack(32);
  v.pushBack(12454);
  if(ASSERT_EQ_INT("push back value", 12454, v.back()))*passed += 1;
}

void test_vector_push_at(unsigned int* total, unsigned int* passed) {
  *total += 6;
  Essentials::Vector<int> v;
  v.setCapacity(20);
  v.pushHead(1); v.pushHead(2); v.pushHead(3); v.pushHead(5); v.pushHead(3);
  v.pushBack(12); v.pushBack(123);
  v.pushAt(1, 77);
  if(ASSERT_EQ_INT("push at 1", 77, v.at(1)))*passed += 1;
  v.pushAt(3, 12);
  if(ASSERT_EQ_INT("push at 3", 12, v.at(3)))*passed += 1;
  v.pushAt(-1, 98);
  if(ASSERT_EQ_INT("push at -1", 3, v.at(-1)))*passed += 1;
  if(ASSERT_EQ_INT("push at -2", 98, v.at(-2)))*passed += 1;
  if(ASSERT_EQ_INT("push at back", 123, v.back()))*passed += 1;
  if(ASSERT_EQ_INT("push at head", 3, v.head()))*passed += 1;
}

// ===== POP METHODS =====
void test_vector_pop_head(unsigned int* total, unsigned int* passed) {
  *total += 2;
  Essentials::Vector<int> v;
  v.pushHead(5); v.pushHead(6);
  if(ASSERT_EQ_INT("pop head", 6, v.popHead()))*passed += 1;
  if(ASSERT_EQ_INT("pop head size", 1, v.getSize()))*passed += 1;
}

void test_vector_pop_back(unsigned int* total, unsigned int* passed) {
  *total += 2;
  Essentials::Vector<int> v;
  v.pushHead(5); v.pushHead(6);
  if(ASSERT_EQ_INT("pop front", 5, v.popBack()))*passed += 1;
  if(ASSERT_EQ_INT("pop back size", 1, v.getSize()))*passed += 1;
}

void test_vector_pop_at(unsigned int* total, unsigned int* passed) {
  *total += 7;
  Essentials::Vector<int> v;
  v.setCapacity(20);
  v.pushHead(1); v.pushHead(2); v.pushHead(3); v.pushHead(5);
  v.pushBack(12); v.pushBack(54);
  if(ASSERT_EQ_INT("pop_at 1", 12, v.popAt(1)))*passed += 1;
  if(ASSERT_EQ_INT("pop back size", 5, v.getSize()))*passed += 1;
  if(ASSERT_EQ_INT("pop_at 2", 2, v.popAt(2)))*passed += 1;
  if(ASSERT_EQ_INT("pop_at -1", 5, v.popAt(-1)))*passed += 1;
  if(ASSERT_EQ_INT("pop back size", 3, v.getSize()))*passed += 1;
  if(ASSERT_EQ_INT("pop at back", 54, v.back()))*passed += 1;
  if(ASSERT_EQ_INT("pop at head", 3, v.head()))*passed += 1;
}

// ===== ERASE & CLEAR =====
void test_vector_erase(unsigned int* total, unsigned int* passed) {
  *total += 3;
  Essentials::Vector<int> v;
  v.setCapacity(10);
  v.pushHead(231); v.pushHead(2); v.pushHead(3);
  v.pushBack(1); v.pushBack(2);
  v.erase(1);
  v.erase(-1);
  if(ASSERT_EQ_INT("erase at 1", 231, v.at(1)))*passed += 1;
  if(ASSERT_EQ_INT("erase at -1", 2, v.at(-1)))*passed += 1;
  if(ASSERT_EQ_INT("pop back size", 3, v.getSize()))*passed += 1;
}

void test_vector_clear(unsigned int* total, unsigned int* passed) {
  *total += 1;
  Essentials::Vector<int> v;
  v.pushHead(1); v.pushHead(5); v.pushHead(3);
  v.pushBack(1); v.pushBack(2);
  v.clear();
  if(ASSERT_EQ_SIZE("clear resets size", 0, v.getSize()))*passed += 1;
}

// ===== Head / Back / AT =====
void test_vector_head_back_at(unsigned int* total, unsigned int* passed) {
  *total += 3;
  Essentials::Vector<int> v;
  v.setCapacity(10);
  v.pushHead(12); v.pushHead(45); v.pushHead(56);
  v.pushBack(32); v.pushBack(213); v.pushBack(512);
  if(ASSERT_EQ_INT("head", 56, v.head()))*passed += 1;
  if(ASSERT_EQ_INT("back", 512, v.back()))*passed += 1;
  if(ASSERT_EQ_INT("at[1]", 213, v.at(1)))*passed += 1;
}

// ===== GETTERS =====
void test_vector_get_size(unsigned int* total, unsigned int* passed) {
  *total += 2;
  Essentials::Vector<int> v;
  if(ASSERT_EQ_SIZE("get_size empty",0,v.getSize()))*passed += 1;
  v.pushHead(5);v.pushHead(5);v.pushHead(5);v.pushHead(5);
  if(ASSERT_EQ_SIZE("get_size=1",4,v.getSize()))*passed += 1;
}

void test_vector_get_capacity(unsigned int* total, unsigned int* passed) {
  *total += 1;
  Essentials::Vector<int> v;
  if(ASSERT_TRUE("get_capacity>=1",v.getCapacity()>=1))*passed += 1;
}

void test_vector_is_empty(unsigned int* total, unsigned int* passed) {
  *total += 2;
  Essentials::Vector<int> v;
  if(ASSERT_TRUE("is_empty true",v.isEmpty()))*passed += 1;
  v.pushHead(3);
  if(ASSERT_TRUE("is_empty false",!v.isEmpty()))*passed += 1;
}

// ===== MIXED PUSH / POP =====
void test_vector_mixed_push_pop(unsigned int* total, unsigned int* passed) {
  *total += 4;
  Essentials::Vector<int> v;
  v.pushHead(1); v.pushBack(2); v.pushHead(3); v.pushBack(4);
  if(ASSERT_EQ_INT("mixed back",4,v.back()))*passed += 1;
  if(ASSERT_EQ_INT("mixed head",3,v.head()))*passed += 1;
  if(ASSERT_EQ_INT("pop Back",4,v.popBack()))*passed += 1;
  if(ASSERT_EQ_INT("pop Head",3,v.popHead()))*passed += 1;
}

// ===== LARGE DATASET TESTS =====
void test_vector_large_push_head(unsigned int* total, unsigned int* passed) {
  *total += 3;
  Essentials::Vector<int> v;
  int N=1000;
  for(int i=0;i<N;i++) v.pushHead(i);
  if(ASSERT_EQ_SIZE("large push size",N,v.getSize()))*passed += 1;
  if(ASSERT_EQ_INT("large push head",N-1,v.head()))*passed += 1;
  if(ASSERT_EQ_INT("large push back",0,v.back()))*passed += 1;
}

void test_vector_large_push_back(unsigned int* total, unsigned int* passed) {
  *total += 3;
  Essentials::Vector<int> v;
  int N=1000;
  for(int i=0;i<N;i++) v.pushBack(i);
  if(ASSERT_EQ_SIZE("large push head size",N,v.getSize()))*passed += 1;
  if(ASSERT_EQ_INT("large push head back",N-1,v. back()))*passed += 1;
  if(ASSERT_EQ_INT("large push head head",0,v.head()))*passed += 1;
}

// ===== RUN ALL =====
std::pair<unsigned int, unsigned int> Vector() {
  unsigned int total = 0;
  unsigned int passed = 0;


  // Init
  test_vector_init(&total, &passed);

  // Resizes
  test_vector_set_capacity(&total, &passed);
  test_vector_reorder(&total, &passed); 
  test_vector_reserve(&total, &passed);
  test_vector_resize(&total, &passed); 
  test_vector_shrink(&total, &passed);
  
  // Getters
  test_vector_get_size(&total, &passed);
  test_vector_get_capacity(&total, &passed);
  test_vector_is_empty(&total, &passed);
  test_vector_head_back_at(&total, &passed);

  // Pushers
  test_vector_push_head(&total, &passed);
  test_vector_push_back(&total, &passed);
  test_vector_push_at(&total, &passed);
  
  // Popers
  test_vector_pop_head(&total, &passed);
  test_vector_pop_back(&total, &passed);
  test_vector_pop_at(&total, &passed);

  // Erasers
  test_vector_erase(&total, &passed);
  test_vector_clear(&total, &passed);
  
  // operators

  // mixed and large
  test_vector_mixed_push_pop(&total, &passed);
  test_vector_large_push_head(&total, &passed);
  test_vector_large_push_back(&total, &passed);


  if(passed == total) 
    fmt::print(fg(fmt::color::blue_violet), "=== Essentials::Vector Test Score: {:d}\\{:d} passed ===\n", passed, total);
  else 
    fmt::print(fg(fmt::color::dark_orange), "=== Essentials::Vector Test Score: {:d}\\{:d} passed ===\n", passed, total);
  
  return std::pair<unsigned int, unsigned int>(passed, total);
};
};