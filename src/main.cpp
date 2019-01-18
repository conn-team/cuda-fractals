#include <assert.h>
#include <iota.hpp>
#include <stdio.h>

int main() {
  int vec[1024];
  iota(1024, vec);

  for (int i = 0; i < 1024; ++i) {
    assert(vec[i] == i);
    printf("%d ", vec[i]);
  }
  printf("\n");
  fflush(stdout);
  return 0;
}