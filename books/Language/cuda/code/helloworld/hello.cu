#include <stdio.h>

 __global__ void greet(){
    printf("Hello World\n");
}

int main(){
    greet<<<1,1>>>();
    return 0;
}