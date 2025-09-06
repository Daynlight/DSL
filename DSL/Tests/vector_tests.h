#ifndef PRAGMA_VECTOR_FULL_TESTS
#define PRAGMA_VECTOR_FULL_TESTS

#include <stdio.h>
#include <string.h>
#include "../DSL/dsl.h"

typedef struct {
    int x;
    int y;
} Point;

void VectorTests() {
    printf("=== Full Vector Tests ===\n");

    int total_tests = 0;
    int passed_tests = 0;

    // ===== INT VECTOR =====
    Vector vi;
    vector_init(&vi, sizeof(int));

    // Test: initial state
    total_tests++;
    if (vector_get_size(&vi) == 0 && vector_get_capacity(&vi) >= 1 && vector_is_empty(&vi)) {
        passed_tests++; printf("[PASS] INT: initial state\n");
    } else printf("[FAIL] INT: initial state\n");

    // Test: push and push_front
    int a = 1, b = 2, c = 3;
    vector_push(&vi, &a);
    vector_push(&vi, &b);
    vector_push_front(&vi, &c);

    int first_val = -1;
    vector_first(&vi, &first_val);
    total_tests++;
    if (first_val == 3 && vector_get_size(&vi) == 3) { passed_tests++; printf("[PASS] INT: push/push_front\n"); }
    else printf("[FAIL] INT: push/push_front\n");

    // Test: push_at
    int d = 99;
    vector_push_at(&vi, 1, &d);
    int check = -1;
    vector_at(&vi, 1, &check);
    total_tests++;
    if (check == 99) { passed_tests++; printf("[PASS] INT: push_at\n"); }
    else printf("[FAIL] INT: push_at\n");

    // Test: pop_at
    int popped = -1;
    vector_pop_at(&vi, 1, &popped);
    total_tests++;
    if (popped == 99 && vector_get_size(&vi) == 3) { passed_tests++; printf("[PASS] INT: pop_at\n"); }
    else printf("[FAIL] INT: pop_at\n");

    // Test: pop_front
    int popped_front = -1;
    vector_pop_front(&vi, &popped_front);
    total_tests++;
    if (popped_front == 3 && vector_get_size(&vi) == 2) { passed_tests++; printf("[PASS] INT: pop_front\n"); }
    else printf("[FAIL] INT: pop_front\n");

    // Test: pop last
    int last_val = -1;
    vector_pop(&vi, &last_val);
    total_tests++;
    if (last_val == 2 && vector_get_size(&vi) == 1) { passed_tests++; printf("[PASS] INT: pop last\n"); }
    else printf("[FAIL] INT: pop last\n");

    // Test: erase
    vector_erase(&vi, 0);
    total_tests++;
    if (vector_get_size(&vi) == 0) { passed_tests++; printf("[PASS] INT: erase\n"); }
    else printf("[FAIL] INT: erase\n");

    // Test: clear
    vector_push(&vi, &a);
    vector_push(&vi, &b);
    vector_clear(&vi);
    total_tests++;
    if (vector_get_size(&vi) == 0) { passed_tests++; printf("[PASS] INT: clear\n"); }
    else printf("[FAIL] INT: clear\n");

    // Test: vector_last
    vector_push(&vi, &a);
    vector_push(&vi, &b);
    int last = -1;
    vector_last(&vi, &last);
    total_tests++;
    if (last == 2) { passed_tests++; printf("[PASS] INT: last\n"); }
    else printf("[FAIL] INT: last\n");

    // Test: vector_at out-of-bounds
    int out = -1;
    vector_at(&vi, 10, &out); // should not change
    total_tests++;
    if (out == -1) { passed_tests++; printf("[PASS] INT: at out-of-bounds\n"); }
    else printf("[FAIL] INT: at out-of-bounds\n");

    // Test: get_size, get_capacity, get_element_size, vector_get_sorted, is_empty
    total_tests++;
    if (vector_get_size(&vi) == 2 &&
        vector_get_capacity(&vi) >= 2 &&
        vector_get_element_size(&vi) == sizeof(int) &&
        vector_get_sorted(&vi) == 0 &&
        !vector_is_empty(&vi)) { passed_tests++; printf("[PASS] INT: getters\n"); }
    else printf("[FAIL] INT: getters\n");

    // Test: resize, reserve, shrink, set_capacity
    unsigned int old_cap = vector_get_capacity(&vi);
    vector_resize(&vi);
    total_tests++;
    if (vector_get_capacity(&vi) >= old_cap * 2) { passed_tests++; printf("[PASS] INT: resize\n"); }
    else printf("[FAIL] INT: resize\n");

    unsigned int cap_before = vector_get_capacity(&vi);
    vector_reserve(&vi, 5);
    total_tests++;
    if (vector_get_capacity(&vi) >= cap_before + 5) { passed_tests++; printf("[PASS] INT: reserve\n"); }
    else printf("[FAIL] INT: reserve\n");

    vector_shrink(&vi);
    total_tests++;
    if (vector_get_capacity(&vi) == vector_get_size(&vi)) { passed_tests++; printf("[PASS] INT: shrink\n"); }
    else printf("[FAIL] INT: shrink\n");

    vector_set_capacity(&vi, 10);
    total_tests++;
    if (vector_get_capacity(&vi) == 10) { passed_tests++; printf("[PASS] INT: set_capacity\n"); }
    else printf("[FAIL] INT: set_capacity\n");

    vector_destroy(&vi);

    // ===== FLOAT VECTOR =====
    Vector vf;
    vector_init(&vf, sizeof(float));
    float f1 = 1.1f, f2 = 2.2f, f3 = 3.3f;
    vector_push(&vf, &f1);
    vector_push(&vf, &f2);
    vector_push_at(&vf, 1, &f3);

    float fcheck = -1;
    vector_at(&vf, 1, &fcheck);
    total_tests++;
    if (fcheck == 3.3f) { passed_tests++; printf("[PASS] FLOAT: push/push_at\n"); }
    else printf("[FAIL] FLOAT: push/push_at\n");

    vector_destroy(&vf);

    // ===== CHAR VECTOR =====
    Vector vc;
    vector_init(&vc, sizeof(char));
    char ch1='a', ch2='b', ch3='c';
    vector_push(&vc, &ch1);
    vector_push(&vc, &ch2);
    vector_push_front(&vc, &ch3);

    char cfirst = 0;
    vector_first(&vc, &cfirst);
    total_tests++;
    if (cfirst == 'c') { passed_tests++; printf("[PASS] CHAR: push/push_front/first\n"); }
    else printf("[FAIL] CHAR: push/push_front/first\n");

    vector_destroy(&vc);

    // ===== STRUCT VECTOR =====
    Vector vp;
    vector_init(&vp, sizeof(Point));
    Point p1 = {1,2}, p2 = {3,4}, p3 = {5,6};
    vector_push(&vp, &p1);
    vector_push_front(&vp, &p2);
    vector_push_at(&vp, 1, &p3);

    Point pcheck;
    vector_at(&vp, 1, &pcheck);
    total_tests++;
    if (pcheck.x == 5 && pcheck.y == 6) { passed_tests++; printf("[PASS] STRUCT: push/push_front/push_at\n"); }
    else printf("[FAIL] STRUCT: push/push_front/push_at\n");

    vector_pop_front(&vp, &pcheck);
    total_tests++;
    if (pcheck.x == 3 && pcheck.y == 4 && vector_get_size(&vp) == 2) { passed_tests++; printf("[PASS] STRUCT: pop_front\n"); }
    else printf("[FAIL] STRUCT: pop_front\n");

    vector_pop_at(&vp, 0, &pcheck);
    total_tests++;
    if (pcheck.x == 5 && pcheck.y == 6 && vector_get_size(&vp) == 1) { passed_tests++; printf("[PASS] STRUCT: pop_at\n"); }
    else printf("[FAIL] STRUCT: pop_at\n");

    vector_clear(&vp);
    total_tests++;
    if (vector_get_size(&vp) == 0) { passed_tests++; printf("[PASS] STRUCT: clear\n"); }
    else printf("[FAIL] STRUCT: clear\n");

    vector_destroy(&vp);

    printf("\n=== Vector Full Test Score: %d/%d passed ===\n", passed_tests, total_tests);
}

#endif
