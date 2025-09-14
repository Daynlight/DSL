#pragma once
#include <iostream>
#include "../DSL/dsl.h"

struct Point { int x, y; };

// ===== Comparators & Predicates =====
int cmp_int(const int &a, const int &b) { return a - b; }
int is_even(const int &val) { return (val % 2) == 0; }
int point_cmp(const Point &a, const Point &b) { return a.x - b.x; }

void add(const int &el, int &ud) { ud += el; }
void multiply(int &el) { el *= 2; }

// ===== ASSERT MACROS =====
#define ASSERT_EQ_INT(name, expected, actual) do { int e=(expected), a=(actual); bool res=(e==a); std::cout << "[" << (res?"PASS":"FAIL") << "] " << name << " | exp=" << e << " got=" << a << std::endl; if(!res) return false; } while(0)
#define ASSERT_EQ_SIZE(name, expected, actual) do { size_t e=(expected), a=(actual); bool res=(e==a); std::cout << "[" << (res?"PASS":"FAIL") << "] " << name << " | exp=" << e << " got=" << a << std::endl; if(!res) return false; } while(0)
#define ASSERT_EQ_FLOAT(name, expected, actual, eps) do { float e=(expected), a=(actual); float diff = std::abs(e-a); bool res=(diff<eps); std::cout << "[" << (res?"PASS":"FAIL") << "] " << name << " | exp=" << e << " got=" << a << std::endl; if(!res) return false; } while(0)
#define ASSERT_TRUE(name, cond) do { bool res=(cond); std::cout << "[" << (res?"PASS":"FAIL") << "] " << name << std::endl; if(!res) return false; } while(0)

// ===== INIT & DESTROY =====
bool test_init() {
    Vector<int> v;
    ASSERT_EQ_SIZE("init size", 0, v.getSize());
    ASSERT_TRUE("init capacity>=1", v.getCapacity()>=1);
    return true;
}

// ===== RESIZE & CAPACITY =====
bool test_resize() {
    Vector<int> v;
    unsigned old = v.getCapacity();
    v.resize();
    ASSERT_TRUE("resize doubles capacity", v.getCapacity()>=old*2);
    return true;
}

bool test_reserve() {
    Vector<int> v;
    unsigned old = v.getCapacity();
    v.reserve(10);
    ASSERT_TRUE("reserve increases capacity", v.getCapacity()>=old+10);
    return true;
}

bool test_set_capacity() {
    Vector<int> v;
    v.setCapacity(20);
    ASSERT_EQ_SIZE("set_capacity=20", 20, v.getCapacity());
    return true;
}

bool test_shrink() {
    Vector<int> v;
    v.pushHead(1);
    v.shrink();
    ASSERT_EQ_SIZE("shrink size==capacity", v.getSize(), v.getCapacity());
    return true;
}

// ===== PUSH METHODS =====
bool test_push() {
    Vector<int> v;
    v.pushHead(42);
    ASSERT_EQ_INT("push value", 42, v.first());
    return true;
}

bool test_push_front() {
    Vector<int> v;
    v.pushHead(1);
    v.pushHead(2);
    v.pushFront(99);
    ASSERT_EQ_INT("push_front value", 99, v.first());
    return true;
}

bool test_push_at() {
    Vector<int> v;
    v.pushHead(1); v.pushHead(2); v.pushHead(3);
    v.pushAt(1, 77);
    ASSERT_EQ_INT("push_at middle", 77, v.at(1));
    return true;
}

// ===== POP METHODS =====
bool test_pop() {
    Vector<int> v;
    v.pushHead(5); v.pushHead(6);
    ASSERT_EQ_INT("pop last", 6, v.popHead());
    return true;
}

bool test_pop_front() {
    Vector<int> v;
    v.pushHead(5); v.pushHead(6);
    ASSERT_EQ_INT("pop front", 5, v.popBack());
    return true;
}

bool test_pop_at() {
    Vector<int> v;
    v.pushHead(1); v.pushHead(2); v.pushHead(3);
    ASSERT_EQ_INT("pop_at middle", 2, v.popAt(1));
    return true;
}

// ===== ERASE & CLEAR =====
bool test_erase() {
    Vector<int> v;
    v.pushHead(1); v.pushHead(2); v.pushHead(3);
    v.erase(1);
    ASSERT_EQ_INT("erase shifts", 3, v.at(1));
    return true;
}

bool test_clear() {
    Vector<int> v;
    v.pushHead(1);
    v.clear();
    ASSERT_EQ_SIZE("clear resets size", 0, v.getSize());
    return true;
}

// ===== FIRST / LAST / AT =====
bool test_first_last_at() {
    Vector<int> v;
    v.pushHead(11); v.pushHead(22); v.pushHead(33);
    ASSERT_EQ_INT("last", 33, v.last());
    ASSERT_EQ_INT("first", 11, v.first());
    ASSERT_EQ_INT("at[1]", 22, v.at(1));
    return true;
}

// // ===== SEARCH METHODS =====
// bool test_find_methods() {
//     Vector<int> v;
//     v.pushHead(1); v.pushHead(2); v.pushHead(3);
//     ASSERT_EQ_SIZE("find 2", 1, v.find(2, cmp_int));
//     ASSERT_EQ_SIZE("find_if even", 1, v.find_if(is_even));
//     ASSERT_TRUE("contains 3", v.contains(3, cmp_int));
//     v.qsort(cmp_int);
//     ASSERT_EQ_SIZE("bfind 2", 1, v.bfind(2, cmp_int));
//     ASSERT_EQ_SIZE("bfind_if even", 1, v.bfind_if(is_even));
//     ASSERT_TRUE("bcontains 3", v.bcontains(3, cmp_int));
//     return true;
// }

// // ===== QSORT =====
// bool test_qsort_variants() {
//     Vector<int> vi; vi.pushHead(30); vi.pushHead(10); vi.pushHead(20);
//     vi.qsort(cmp_int);
//     ASSERT_EQ_INT("qsort int first", 10, vi.first());

//     Vector<float> vf; vf.pushHead(3.3f); vf.pushHead(1.1f); vf.pushHead(2.2f);
//     vf.qsort([](const float &a,const float &b){return (a>b)-(a<b);});
//     ASSERT_EQ_FLOAT("qsort float first", 1.1f, vf.first(), 1e-6f);

//     Vector<Point> vs; vs.pushHead({2,2}); vs.pushHead({1,1}); vs.pushHead({3,3});
//     vs.qsort(point_cmp);
//     ASSERT_EQ_INT("qsort struct first x",1, vs.first().x);

//     return true;
// }

// // ===== MAP / FOREACH =====
// bool test_map_foreach() {
//     Vector<int> v; v.pushHead(1); v.pushHead(2); v.pushHead(3);
//     v.map(multiply);
//     int sum=0;
//     v.foreach([&](int el){ add(el,sum); });
//     ASSERT_EQ_INT("map*2 + foreach sum",12,sum);
//     return true;
// }

// // ===== SWAP =====
// bool test_swap() {
//     Vector<int> a,b; a.pushHead(1); b.pushHead(2);
//     a.swap(b);
//     ASSERT_EQ_INT("swap a",2,a.first());
//     ASSERT_EQ_INT("swap b",1,b.first());
//     return true;
// }

// ===== GETTERS =====
bool test_get_size() {
    Vector<int> v;
    ASSERT_EQ_SIZE("get_size empty",0,v.getSize());
    v.pushHead(5);
    ASSERT_EQ_SIZE("get_size=1",1,v.getSize());
    return true;
}

bool test_get_capacity() {
    Vector<int> v;
    ASSERT_TRUE("get_capacity>=1",v.getCapacity()>=1);
    return true;
}

// bool test_get_sorted() {
//     Vector<int> v;
//     ASSERT_EQ_INT("get_sorted init",1,v.getSorted());
//     v.pushHead(2);
//     ASSERT_EQ_INT("get_sorted after push",0,v.getSorted());
//     v.qsort(cmp_int);
//     ASSERT_EQ_INT("get_sorted after sort",1,v.getSorted());
//     return true;
// }

bool test_is_empty() {
    Vector<int> v;
    ASSERT_TRUE("is_empty true",v.isEmpty());
    v.pushHead(3);
    ASSERT_TRUE("is_empty false",!v.isEmpty());
    return true;
}

// // ===== CONTAINS =====
// bool test_contains_unsorted() {
//     Vector<int> v; v.pushHead(3); v.pushHead(1); v.pushHead(2);
//     ASSERT_TRUE("contains 1",v.contains(1, cmp_int));
//     ASSERT_TRUE("contains 99 false", !v.contains(99, cmp_int));
//     return true;
// }

// bool test_contains_sorted() {
//     Vector<int> v; v.pushHead(1); v.pushHead(2); v.pushHead(3);
//     v.qsort(cmp_int);
//     ASSERT_TRUE("bcontains sorted ok",v.contains(2, cmp_int));
//     return true;
// }

// bool test_bcontains() {
//     Vector<int> v; v.pushHead(10); v.pushHead(20); v.pushHead(30);
//     v.qsort(cmp_int);
//     ASSERT_TRUE("bcontains 20",v.bcontains(20, cmp_int));
//     ASSERT_TRUE("bcontains missing",!v.bcontains(100, cmp_int));
//     return true;
// }

// ===== MIXED PUSH / POP =====
bool test_mixed_push_pop() {
    Vector<int> v;
    v.pushHead(1); v.pushFront(2); v.pushHead(3); v.pushFront(4);
    ASSERT_EQ_INT("mixed first",4,v.first());
    ASSERT_EQ_INT("mixed last",3,v.last());
    ASSERT_EQ_INT("pop_front",4,v.popBack());
    ASSERT_EQ_INT("pop_back",3,v.popHead());
    return true;
}

// bool test_mixed_algorithms() {
//     Vector<int> v;
//     for(int i=0;i<10;i++) {
//         if(i%2==0) v.pushFront(i);
//         else v.pushHead(i);
//     }
//     v.qsort(cmp_int);
//     ASSERT_EQ_INT("mixed sort first",0,v.first());
//     ASSERT_EQ_INT("mixed sort last",9,v.last());
//     ASSERT_TRUE("mixed contains 6", v.bcontains(6, cmp_int));
//     ASSERT_TRUE("mixed contains missing", !v.bcontains(42, cmp_int));
//     int sum=0; v.map(multiply); v.foreach([&](int el){ add(el,sum); });
//     int expected=90; // sum 0..9 doubled
//     ASSERT_EQ_INT("mixed map+foreach sum",expected,sum);
//     return true;
// }

// ===== LARGE DATASET TESTS =====
bool test_large_push() {
    Vector<int> v;
    int N=1000;
    for(int i=0;i<N;i++) v.pushHead(i);
    ASSERT_EQ_SIZE("large push size",N,v.getSize());
    ASSERT_EQ_INT("large push last",N-1,v.last());
    ASSERT_EQ_INT("large push first",0,v.first());
    return true;
}

bool test_large_push_front() {
    Vector<int> v;
    int N=1000;
    for(int i=0;i<N;i++) v.pushFront(i);
    ASSERT_EQ_SIZE("large push_front size",N,v.getSize());
    ASSERT_EQ_INT("large push_front first",N-1,v.first());
    ASSERT_EQ_INT("large push_front last",0,v.last());
    return true;
}

// ... (other large tests: large_qsort, large_find_bfind, large_contains, large_struct_qsort, large_map_foreach, mixed_large) ...

// ===== RUN ALL =====
void VectorTests() {
    int total=0, passed=0;
#define RUN(f) total++; if(f()) passed++;

    RUN(test_init);
    RUN(test_resize);
    RUN(test_reserve);
    RUN(test_set_capacity);
    RUN(test_shrink);
    RUN(test_push);
    RUN(test_push_front);
    RUN(test_push_at);
    RUN(test_pop);
    RUN(test_pop_front);
    RUN(test_pop_at);
    RUN(test_erase);
    RUN(test_clear);
    RUN(test_first_last_at);
    RUN(test_get_size);
    RUN(test_get_capacity);
    RUN(test_is_empty);
    RUN(test_mixed_push_pop);
    // RUN(test_find_methods);
    // RUN(test_qsort_variants);
    // RUN(test_map_foreach);
    // RUN(test_swap);
    // RUN(test_get_sorted);
    // RUN(test_contains_unsorted);
    // RUN(test_contains_sorted);
    // RUN(test_bcontains);
    // RUN(test_mixed_algorithms);

    RUN(test_large_push);
    RUN(test_large_push_front);

#undef RUN
    std::cout << "\n=== Vector Test Score: " << passed << "/" << total << " passed ===\n";
}
