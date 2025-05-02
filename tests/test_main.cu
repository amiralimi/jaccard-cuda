#include <iostream>

void test_reduce_sum(int seed = 42);
void test_intersection_union(int seed = 42);
void test_jaccard_similarity(int seed = 42);
void test_compress_1bit(int seed = 42);

int main()
{
    // test_reduce_sum(42);
    // test_intersection_union(42);
    // test_jaccard_similarity(42);
    test_compress_1bit(42);

    std::cout << "\nAll requested unit‑tests passed\n";
    return 0;
}
