#ifndef PRAGMA_VECTOR_EXAMPLE
#define PRAGMA_VECTOR_EXAMPLE
#include "../DSL/dsl.h"

void VectorExample(){
  Vector vec;
  vector_init(&vec, sizeof(int));

  int x = 10;
  vector_push(&vec, &x);

  x = 20;
  vector_push(&vec, &x);

  int value;
  vector_at(&vec, 1, &value);
  printf("Second element: %d\n", value);

  vector_destroy(&vec);
};

#endif