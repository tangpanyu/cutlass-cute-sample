#include <cuda.h>
#include <stdlib.h>

#include "util.h"

using namespace cute;

int main()
{
    auto layout_a = make_layout(make_shape(Int<4>{}, Int<4>{}),
                                make_stride(Int<4>{}, Int<1>{}));
    auto layout_a_inv = left_inverse(layout_a);
    auto inv_then_a = composition(layout_a_inv, layout_a);

    // print_latex(layout_a);
    print("\n");

    print_latex(layout_a_inv);
    print("\n");

    // print_latex(inv_then_a);
    print("\n");
}
