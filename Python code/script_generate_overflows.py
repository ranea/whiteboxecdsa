"""Old method to generate the list of overflows.

Generate all 5-bit binary numbers with increasing hamming weight, 
add them 7 times, and store the overflows in a list for the C source.
"""

import itertools

import sage.all


def next_combination(x):
    u = x & (-x)
    v = u + x
    return v + ((v.__xor__(x) // u) >> 2)


def get_binary_numbers_increasing_hw(bitsize):
    index_list = 0
    my_list = [[]]
    for k in range(1, bitsize + 1):  # k = number of 1's
        x = (1 << k) - 1
        my_list[index_list].append(int2vector(int(x), bitsize))
        while True:
            x = next_combination(x)
            if (x >> bitsize) != 0:
                index_list += 1
                my_list.append([])
                break
            else:
                my_list[index_list].append(int2vector(int(x), bitsize))
    # assert set(my_list) == set(range(2**bitsize))
    return my_list[:-1]


def int2vector(x, size):
    v = bin(x)[2:][::-1]
    v = v + "0" * (size - len(v))
    aux = sage.all.vector(sage.all.ZZ, [int(v_i) for v_i in v])
    aux.set_immutable()
    return aux


def vector2int(v):
    return int("0b" + "".join([str(v_i) for v_i in reversed(v)]), base=2)


def get_list_overflows(bitsize, repetitions, verbose=False):
    list_bin_per_hw = get_binary_numbers_increasing_hw(bitsize)
    list_overflows = [int2vector(int(0), bitsize)] + sage.all.flatten(list_bin_per_hw)
    set_list_overflows = set(list_overflows)
    if verbose:
        print(list_bin_per_hw)
        print(list_overflows)
        print()

    start_index_copy = 1
    for index_repetition in range(repetitions - 1):
        if verbose:
            print("index_repetition:", index_repetition)
        copy_list_overflows = list_overflows[start_index_copy:]
        start_index_copy = len(list_overflows)
        for list_bin_fixed_hw in list_bin_per_hw:
            for o in copy_list_overflows:
                if verbose: print(f"{o} + {list_bin_fixed_hw}")
                for bin in list_bin_fixed_hw:
                    new_o = o + bin
                    new_o.set_immutable()
                    if new_o not in set_list_overflows:
                        list_overflows.append(new_o)
                        set_list_overflows.add(new_o)

    aux = list(itertools.product(range(repetitions + 1), repeat=bitsize))
    assert set(aux) == set(tuple([int(o_i) for o_i in o]) for o in list_overflows)
    if verbose: print("num overflows:", len(list_overflows), len(aux))

    list_overflows = [sage.all.vector(sage.all.ZZ, [repetitions for _ in range(bitsize)]) - o for o in list_overflows]

    if verbose:
        print(list_overflows)

    return list_overflows


if __name__ == '__main__':   
    FAST_MODE = None  # True or False

    if FAST_MODE:
        bitsize = int(3)  # number of variables
        from implementation_fast import MAX_ROW_WEIGHT
        assert MAX_ROW_WEIGHT == 4
    else:
        bitsize = int(5)  # number of variables
        # repetitions = int(6 + 1)  # maximum weight
        from implementation_slow import MAX_ROW_WEIGHT
        assert MAX_ROW_WEIGHT == 7
    verbose = False
    list_overflows = get_list_overflows(bitsize, (int(MAX_ROW_WEIGHT) - 1) + 1, verbose)

    import os

    def get_smart_print(filename):
        if filename is None:
            def smart_print(*args):
                print(*args, end="")
        else:
            def smart_print(*args):
                with open(filename, "a") as f:
                    print(*args, file=f, end="")
        return smart_print

    if FAST_MODE:
        filename = "list_overflows_fast.c"  # None
    else:
        filename = "list_overflows.c"  # None
    if filename is not None:
        assert not os.path.isfile(filename), f"{filename} already exists"

    smart_print = get_smart_print(filename)

    smart_print(f'const unsigned short list_overflows[{len(list_overflows)}] = {{')
    for index_overflow, overflow in enumerate(list_overflows):
        x = overflow
        modulus = ((MAX_ROW_WEIGHT - 1) + 1) + 1  # MAX_ROW_WEIGHT + 1
        if FAST_MODE:
            assert bitsize == 3
            enc = ((((x[2]) * modulus) + x[1]) * modulus) + x[0]
        else:
            assert bitsize == 5
            enc = (((((((x[4] * modulus) + x[3]) * modulus) + x[2]) * modulus) + x[1]) * modulus) + x[0]

        if index_overflow == len(list_overflows) - 1:
            smart_print(f'{enc}')
        else:
            smart_print(f'{enc},')
    smart_print('};\n')
