#ifndef PragmaVectorSeparatedTests
#define PragmaVectorSeparatedTests  
#include <stdio.h>
#include <string.h>
#include "../DSL/dsl.h"

typedef struct { int x, y; } Point;

// ===== Comparators & Predicates =====
int cmp_int(const void *a, const void *b) { return *(int*)a - *(int*)b; }
int is_even(const void *val) { return (*(int*)val % 2) == 0; }
int point_cmp(const void *a, const void *b) { return ((Point*)a)->x - ((Point*)b)->x; }
int float_cmp(const void *a, const void *b) { float fa=*(float*)a, fb=*(float*)b; return (fa>fb)-(fa<fb); }
void add(void *el, void *ud) { *(int*)ud += *(int*)el; };
void multiply(void *el) { *(int*)el *= 2; };
// ===== ASSERT MACROS =====
#define ASSERT_EQ_INT(name, expected, actual) do { int e=(expected), a=(actual); int res=(e==a); printf("[%s] %s | exp=%d got=%d\n", res?"PASS":"FAIL", name, e, a); if(!res) return 0; } while(0)
#define ASSERT_EQ_SIZE(name, expected, actual) do { size_t e=(expected), a=(actual); int res=(e==a); printf("[%s] %s | exp=%zu got=%zu\n", res?"PASS":"FAIL", name, e, a); if(!res) return 0; } while(0)
#define ASSERT_EQ_FLOAT(name, expected, actual, eps) do { float e=(expected), a=(actual), diff=e-a; if(diff<0) diff=-diff; int res=diff<(eps); printf("[%s] %s | exp=%.6f got=%.6f\n", res?"PASS":"FAIL", name, e, a); if(!res) return 0; } while(0)
#define ASSERT_TRUE(name, cond) do { int res=(cond); printf("[%s] %s\n", res?"PASS":"FAIL", name); if(!res) return 0; } while(0)

// ===== INIT & DESTROY =====
int test_init() {
    Vector v; vector_init(&v,sizeof(int));
    ASSERT_EQ_SIZE("init size",0,vector_get_size(&v));
    ASSERT_TRUE("init capacity>=1",vector_get_capacity(&v)>=1);
    vector_destroy(&v);
    ASSERT_EQ_SIZE("destroy capacity=0",0,vector_get_capacity(&v));
    return 1;
}

// ===== RESIZE & CAPACITY =====
int test_resize() {
    Vector v; vector_init(&v,sizeof(int));
    unsigned old = vector_get_capacity(&v);
    vector_resize(&v);
    ASSERT_TRUE("resize doubles capacity",vector_get_capacity(&v)>=old*2);
    vector_destroy(&v);
    return 1;
}

int test_reserve() {
    Vector v; vector_init(&v,sizeof(int));
    unsigned old = vector_get_capacity(&v);
    vector_reserve(&v,10);
    ASSERT_TRUE("reserve increases capacity",vector_get_capacity(&v)>=old+10);
    vector_destroy(&v);
    return 1;
}

int test_set_capacity() {
    Vector v; vector_init(&v,sizeof(int));
    vector_set_capacity(&v,20);
    ASSERT_EQ_SIZE("set_capacity=20",20,vector_get_capacity(&v));
    vector_destroy(&v);
    return 1;
}

int test_shrink() {
    Vector v; vector_init(&v,sizeof(int));
    int x=1; vector_push(&v,&x);
    vector_shrink(&v);
    ASSERT_EQ_SIZE("shrink size==capacity",vector_get_size(&v),vector_get_capacity(&v));
    vector_destroy(&v);
    return 1;
}

// ===== PUSH METHODS =====
int test_push() {
    Vector v; vector_init(&v,sizeof(int));
    int x=42; vector_push(&v,&x);
    int out=-1; vector_at(&v,0,&out);
    ASSERT_EQ_INT("push value",42,out);
    vector_destroy(&v);
    return 1;
}

int test_push_front() {
    Vector v; vector_init(&v,sizeof(int));
    int a=1,b=2; vector_push(&v,&a); vector_push(&v,&b);
    int x=99; vector_push_front(&v,&x);
    int out=-1; vector_first(&v,&out);
    ASSERT_EQ_INT("push_front value",99,out);
    vector_destroy(&v);
    return 1;
}

int test_push_at() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={1,2,3}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    int x=77; vector_push_at(&v,1,&x);
    int out=-1; vector_at(&v,1,&out);
    ASSERT_EQ_INT("push_at middle",77,out);
    vector_destroy(&v);
    return 1;
}

// ===== POP METHODS =====
int test_pop() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={5,6}; for(int i=0;i<2;i++) vector_push(&v,&arr[i]);
    int out=-1; vector_pop(&v,&out);
    ASSERT_EQ_INT("pop last",6,out);
    vector_destroy(&v);
    return 1;
}

int test_pop_front() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={5,6}; for(int i=0;i<2;i++) vector_push(&v,&arr[i]);
    int out=-1; vector_pop_front(&v,&out);
    ASSERT_EQ_INT("pop front",5,out);
    vector_destroy(&v);
    return 1;
}

int test_pop_at() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={1,2,3}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    int out=-1; vector_pop_at(&v,1,&out);
    ASSERT_EQ_INT("pop_at middle",2,out);
    vector_destroy(&v);
    return 1;
}

// ===== ERASE & CLEAR =====
int test_erase() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={1,2,3}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    vector_erase(&v,1);
    int out=-1; vector_at(&v,1,&out);
    ASSERT_EQ_INT("erase shifts",3,out);
    vector_destroy(&v);
    return 1;
}

int test_clear() {
    Vector v; vector_init(&v,sizeof(int));
    int x=1; vector_push(&v,&x);
    vector_clear(&v);
    ASSERT_EQ_SIZE("clear resets size",0,vector_get_size(&v));
    vector_destroy(&v);
    return 1;
}

// ===== FIRST / LAST / AT =====
int test_first_last_at() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={11,22,33}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    int f=-1,l=-1,m=-1;
    vector_first(&v,&f); vector_last(&v,&l); vector_at(&v,1,&m);
    ASSERT_EQ_INT("first",11,f);
    ASSERT_EQ_INT("last",33,l);
    ASSERT_EQ_INT("at[1]",22,m);
    vector_destroy(&v);
    return 1;
}

// ===== SEARCH METHODS =====
int test_find_methods() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={1,2,3}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    ASSERT_EQ_SIZE("find 2",1,vector_find(&v,&arr[1],cmp_int));
    ASSERT_EQ_SIZE("find_if even",1,vector_find_if(&v,is_even));
    ASSERT_TRUE("contains 3",vector_contains(&v,&arr[2],cmp_int));
    vector_qsort(&v,cmp_int);
    ASSERT_EQ_SIZE("bfind 2",1,vector_bfind(&v,&arr[1],cmp_int));
    ASSERT_EQ_SIZE("bfind_if even",1,vector_bfind_if(&v,is_even));
    ASSERT_TRUE("bcontains 3",vector_bcontains(&v,&arr[2],cmp_int));
    vector_destroy(&v);
    return 1;
}

// ===== QSORT =====
int test_qsort_variants() {
    Vector vi; vector_init(&vi,sizeof(int));
    int xi[]={30,10,20}; for(int i=0;i<3;i++) vector_push(&vi,&xi[i]);
    vector_qsort(&vi,cmp_int); int f=-1; vector_first(&vi,&f);
    ASSERT_EQ_INT("qsort int first",10,f);
    vector_destroy(&vi);

    Vector vf; vector_init(&vf,sizeof(float));
    float xf[]={3.3f,1.1f,2.2f}; for(int i=0;i<3;i++) vector_push(&vf,&xf[i]);
    vector_qsort(&vf,float_cmp); float fy=-1; vector_first(&vf,&fy);
    ASSERT_EQ_FLOAT("qsort float first",1.1f,fy,1e-6f);
    vector_destroy(&vf);

    Vector vs; vector_init(&vs,sizeof(Point));
    Point pts[]={{2,2},{1,1},{3,3}}; for(int i=0;i<3;i++) vector_push(&vs,&pts[i]);
    vector_qsort(&vs,point_cmp); Point fp; vector_first(&vs,&fp);
    ASSERT_EQ_INT("qsort struct first x",1,fp.x);
    vector_destroy(&vs);
    return 1;
}

// ===== MAP / FOREACH =====
int test_map_foreach() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={1,2,3}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    vector_map(&v,multiply);
    int sum=0;
    vector_foreach(&v,add,&sum);
    ASSERT_EQ_INT("map*2 + foreach sum",12,sum);
    vector_destroy(&v);
    return 1;
}

// ===== SWAP =====
int test_swap() {
    Vector a,b; vector_init(&a,sizeof(int)); vector_init(&b,sizeof(int));
    int x=1,y=2; vector_push(&a,&x); vector_push(&b,&y);
    vector_swap(&a,&b); int out=-1;
    vector_first(&a,&out); ASSERT_EQ_INT("swap a",2,out);
    vector_first(&b,&out); ASSERT_EQ_INT("swap b",1,out);
    vector_destroy(&a); vector_destroy(&b);
    return 1;
}

// ===== GETTERS =====
int test_get_size() {
    Vector v; vector_init(&v,sizeof(int));
    ASSERT_EQ_SIZE("get_size empty",0,vector_get_size(&v));
    int x=5; vector_push(&v,&x);
    ASSERT_EQ_SIZE("get_size=1",1,vector_get_size(&v));
    vector_destroy(&v);
    return 1;
}

int test_get_capacity() {
    Vector v; vector_init(&v,sizeof(int));
    ASSERT_TRUE("get_capacity>=1",vector_get_capacity(&v)>=1);
    vector_destroy(&v);
    return 1;
}

int test_get_element_size() {
    Vector v; vector_init(&v,sizeof(double));
    ASSERT_EQ_SIZE("get_element_size",sizeof(double),vector_get_element_size(&v));
    vector_destroy(&v);
    return 1;
}

int test_get_sorted() {
    Vector v; vector_init(&v,sizeof(int));
    ASSERT_EQ_INT("get_sorted init",1,vector_get_sorted(&v));
    int a=2; vector_push(&v,&a);
    ASSERT_EQ_INT("get_sorted after push",0,vector_get_sorted(&v));
    vector_qsort(&v,cmp_int);
    ASSERT_EQ_INT("get_sorted after sort",1,vector_get_sorted(&v));
    vector_destroy(&v);
    return 1;
}

int test_is_empty() {
    Vector v; vector_init(&v,sizeof(int));
    ASSERT_EQ_INT("is_empty true",1,vector_is_empty(&v));
    int a=3; vector_push(&v,&a);
    ASSERT_EQ_INT("is_empty false",0,vector_is_empty(&v));
    vector_destroy(&v);
    return 1;
}

// ===== CONTAINS =====
int test_contains_unsorted() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={3,1,2}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    ASSERT_TRUE("contains 1",vector_contains(&v,&arr[1],cmp_int));
    ASSERT_TRUE("contains 99 false",!vector_contains(&v,&(int){99},cmp_int));
    vector_destroy(&v);
    return 1;
}

int test_contains_sorted() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={1,2,3}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    vector_qsort(&v,cmp_int);
    ASSERT_TRUE("bcontains sorted ok",vector_contains(&v,&arr[1],cmp_int));
    vector_destroy(&v);
    return 1;
}

int test_bcontains() {
    Vector v; vector_init(&v,sizeof(int));
    int arr[]={10,20,30}; for(int i=0;i<3;i++) vector_push(&v,&arr[i]);
    vector_qsort(&v,cmp_int);
    ASSERT_TRUE("bcontains 20",vector_bcontains(&v,&arr[1],cmp_int));
    ASSERT_TRUE("bcontains missing",!vector_bcontains(&v,&(int){100},cmp_int));
    vector_destroy(&v);
    return 1;
}

// ===== Mixed Tests ===
int test_mixed_push_pop() {
    Vector v; vector_init(&v,sizeof(int));
    int a=1,b=2,c=3,d=4;
    vector_push(&v,&a);   // [1]
    vector_push_front(&v,&b);  // [2,1]
    vector_push(&v,&c);   // [2,1,3]
    vector_push_front(&v,&d);  // [4,2,1,3]

    int first=-1,last=-1;
    vector_first(&v,&first); vector_last(&v,&last);
    ASSERT_EQ_INT("mixed first",4,first);
    ASSERT_EQ_INT("mixed last",3,last);

    int out=-1;
    vector_pop_front(&v,&out); ASSERT_EQ_INT("pop_front",4,out);
    vector_pop(&v,&out);  ASSERT_EQ_INT("pop_back",3,out);

    vector_destroy(&v);
    return 1;
}

int test_mixed_algorithms() {
    Vector v; vector_init(&v,sizeof(int));
    for(int i=0;i<10;i++) {
        if(i%2==0) vector_push_front(&v,&i); // even front
        else       vector_push(&v,&i);  // odd back
    }
    // Now v has evens descending at front, odds ascending at back.

    vector_qsort(&v,cmp_int);

    int first=-1,last=-1;
    vector_first(&v,&first); vector_last(&v,&last);
    ASSERT_EQ_INT("mixed sort first",0,first);
    ASSERT_EQ_INT("mixed sort last",9,last);

    ASSERT_TRUE("mixed contains 6", vector_bcontains(&v,&(int){6},cmp_int));
    ASSERT_TRUE("mixed contains missing", !vector_bcontains(&v,&(int){42},cmp_int));

    int sum=0; vector_map(&v,multiply); vector_foreach(&v,add,&sum);
    int expected = 2* (10*9/2); // sum of 0..9 doubled
    ASSERT_EQ_INT("mixed map+foreach sum",expected,sum);

    vector_destroy(&v);
    return 1;
}


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
    RUN(test_find_methods);
    RUN(test_qsort_variants);
    RUN(test_map_foreach);
    RUN(test_swap);
    RUN(test_get_size);
    RUN(test_get_capacity);
    RUN(test_get_element_size);
    RUN(test_get_sorted);
    RUN(test_contains_unsorted);
    RUN(test_bcontains);
    RUN(test_contains_sorted);
    RUN(test_mixed_push_pop);
    RUN(test_mixed_algorithms);

    #undef RUN
    printf("\n=== Vector Separated Test Score: %d/%d passed ===\n", passed,total);
}



















// ===== LARGE DATASET TESTS =====
int test_large_push() {
    Vector v; vector_init(&v,sizeof(int));
    int N = 1000;
    for(int i=0;i<N;i++) vector_push(&v,&i);
    ASSERT_EQ_SIZE("large push size", N, vector_get_size(&v));
    int first=-1, last=-1;
    vector_first(&v,&first); vector_last(&v,&last);
    ASSERT_EQ_INT("large push first",0,first);
    ASSERT_EQ_INT("large push last", N-1, last);
    vector_destroy(&v);
    return 1;
}

int test_large_push_front() {
    Vector v; vector_init(&v,sizeof(int));
    int N = 1000;
    for(int i=0;i<N;i++) vector_push_front(&v,&i);
    ASSERT_EQ_SIZE("large push_front size", N, vector_get_size(&v));
    int first=-1, last=-1;
    vector_first(&v,&first); vector_last(&v,&last);
    ASSERT_EQ_INT("large push_front first", N-1, first);
    ASSERT_EQ_INT("large push_front last", 0, last);
    vector_destroy(&v);
    return 1;
}

int test_large_qsort() {
    Vector v; vector_init(&v,sizeof(int));
    int N = 1000;
    for(int i=N-1;i>=0;i--) vector_push(&v,&i);
    vector_qsort(&v,cmp_int);
    int first=-1, last=-1;
    vector_first(&v,&first); vector_last(&v,&last);
    ASSERT_EQ_INT("large qsort first",0,first);
    ASSERT_EQ_INT("large qsort last", N-1,last);
    vector_destroy(&v);
    return 1;
}

int test_large_find_bfind() {
    Vector v; vector_init(&v,sizeof(int));
    int N = 1000;
    for(int i=0;i<N;i++) vector_push(&v,&i);
    vector_qsort(&v,cmp_int);
    for(int i=0;i<N;i+=100) {
        unsigned idx = vector_bfind(&v,&i,cmp_int);
        ASSERT_EQ_SIZE("large bfind", i, idx);
    }
    vector_destroy(&v);
    return 1;
}

int test_large_contains() {
    Vector v; vector_init(&v,sizeof(int));
    int N = 1000;
    for(int i=0;i<N;i++) vector_push(&v,&i);
    vector_qsort(&v,cmp_int);
    for(int i=0;i<N;i+=50) ASSERT_TRUE("large bcontains", vector_bcontains(&v,&i,cmp_int));
    ASSERT_TRUE("large bcontains missing", !vector_bcontains(&v,&(int){N+10},cmp_int));
    vector_destroy(&v);
    return 1;
}

int test_large_struct_qsort() {
    Vector v; vector_init(&v,sizeof(Point));
    int N = 500;
    for(int i=N-1;i>=0;i--) {
        Point p = {i, i*2};
        vector_push(&v,&p);
    }
    vector_qsort(&v,point_cmp);
    Point first, last;
    vector_first(&v,&first); vector_last(&v,&last);
    ASSERT_EQ_INT("large struct qsort first.x",0,first.x);
    ASSERT_EQ_INT("large struct qsort last.x",N-1,last.x);
    vector_destroy(&v);
    return 1;
}

int test_large_map_foreach() {
    Vector v; vector_init(&v,sizeof(int));
    int N=1000;
    for(int i=0;i<N;i++) vector_push(&v,&i);
    vector_map(&v,multiply);
    int sum=0;
    vector_foreach(&v,add,&sum);
    int expected = N*(N-1); // sum of 0..N-1 doubled = N*(N-1)
    ASSERT_EQ_INT("large map+foreach sum",expected,sum);
    vector_destroy(&v);
    return 1;
}

int test_mixed_large() {
    Vector v; vector_init(&v,sizeof(int));
    int N=1000;
    for(int i=0;i<N;i++) {
        if(i%2==0) vector_push_front(&v,&i);
        else       vector_push(&v,&i);
    }
    vector_qsort(&v,cmp_int);

    int first=-1,last=-1;
    vector_first(&v,&first); vector_last(&v,&last);
    ASSERT_EQ_INT("mixed large first",0,first);
    ASSERT_EQ_INT("mixed large last",N-1,last);

    for(int i=0;i<N;i+=100) {
        ASSERT_TRUE("mixed large bcontains", vector_bcontains(&v,&i,cmp_int));
    }

    vector_destroy(&v);
    return 1;
}

// ===== RUN LARGE DATASET TESTS =====
void VectorLargeTests() {
    int total=0, passed=0;
    #define RUN(f) total++; if(f()) passed++;

    RUN(test_large_push);
    RUN(test_large_push_front);
    RUN(test_large_qsort);
    RUN(test_large_find_bfind);
    RUN(test_large_contains);
    RUN(test_large_struct_qsort);
    RUN(test_large_map_foreach);
    RUN(test_mixed_large);
    RUN(test_large_map_foreach);

    #undef RUN
    printf("\n=== Vector Large Dataset Test Score: %d/%d passed ===\n", passed,total);
}


#endif
