import functools
import warnings

FAST_MODE = None  # True or False

ENCODING_MODE = "bin_full"  # "bin", "bin_full", "hex"
assert isinstance(ENCODING_MODE, str)

if FAST_MODE:
    from implementation_fast import *
else:
    warnings.warn("importing from implementation_slow")
    from implementation_slow import *

from sage.all import (
    prod
)

# CONSTANTS

if FAST_MODE:
    filename = f"equations_fast_{ENCODING_MODE}.c"  # None
    info_filename = f"info_equations_fast_{ENCODING_MODE}.txt"  # None
else:
    filename = f"equations_{ENCODING_MODE}.c"  # None
    info_filename = f"info_equations_{ENCODING_MODE}.txt"  # None
if filename is not None:
    assert not os.path.isfile(filename), f"{filename} already exists"
if info_filename is not None:
    assert not os.path.isfile(info_filename), f"{info_filename} already exists"


# FUNCTIONS

def get_smart_print(filename, byte_mode=False, encoding_mode=False):
    """Return a print-like function.

    end="" by default.
    """
    if byte_mode:
        r"""
        https://en.wikipedia.org/wiki/Escape_sequences_in_C#Table_of_escape_sequences
        A hex escape sequence must have at least one hex digit following \x, with no upper bound; 
        it continues for as many hex digits as there are.
        
        An octal escape sequence consists of \ followed by one, two, or three octal digits. 
        The octal escape sequence ends when it either contains three octal digits already, or the next character is not an octal digit. 
        In order to denote the byte with numerical value 1, followed by the digit 1, one could use "\1""1", since C automatically concatenates adjacent string literals. 
        The escape sequence \0 is a commonly used octal escape sequence, which denotes the null character, with value zero. 
        
        https://en.cppreference.com/w/c/language/string_literal
        A string literal cannot contain ", \, and newline.
        char* a = "hello" is a string literal, char a[5] = "hello" is not
        
        See also:
        - https://en.cppreference.com/w/c/language/escape
        - https://stackoverflow.com/questions/12208795/c-string-literal-required-escape-characters
        - https://stackoverflow.com/questions/45612822/how-to-properly-add-hex-escapes-into-a-string-literal
        """
        if encoding_mode == "hex":
            replacements = {i: f"\\x{i:0x}" for i in range(256)}
        elif encoding_mode == "bin":
            replacements = {
                0x00: r'\000',  # (avoid warning null character)
                # 0x07: r'\a',
                # 0x08: r'\b',
                # 0x1B: r'\e',
                # 0x0C: r'\f',
                0x0A: r'\n',  # needed
                0x0D: r'\r',  # needed
                # 0x09: r'\t',
                # 0x0B: r'\v',
                0x5C: r'\\',  # needed
                # 0x27: r'\'',
                0x22: r'\"',  # needed
                0x3F: r'\?',  # optionally, to avoid trigraph pragma
            }
        elif encoding_mode == "bin_full":
            replacements = {
                0x0A: r'\n',  # needed
                0x0D: r'\r',  # needed
                0x5C: r'\\',  # needed
                0x22: r'\"',  # needed
            }

        def smart_print(*args):
            assert len(args) == 1
            for c in args[0]:
                if int(c) in replacements:
                    with open(filename, "a") as f:
                        f.write(replacements[int(c)])
                else:
                    with open(filename, "ab") as f:
                        f.write(c.to_bytes(length=1, byteorder='big'))
    else:
        if filename is None:
            def smart_print(*args):
                print(*args, end="")
        else:
            def smart_print(*args):
                with open(filename, "a") as f:
                    print(*args, file=f, end="")
    return smart_print


def int2binary(integer, num_bytes):
    r"""Encode an integer into binary (with non-printable chars).

    This functions returns the bytes of the given integer,
    from the MSB to the LSB.

        >>> int2binary(0, 32)
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        >>> int2binary(1, 32)
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01'
        >>> int2binary(P - 1, 32)
        b'\xff\xff\xff\xff\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xfe'
        >>> int2binary(N - 1, 32)
        b'\xff\xff\xff\xff\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xbc\xe6\xfa\xad\xa7\x17\x9e\x84\xf3\xb9\xca\xc2\xfcc%P'

    """
    from math import ceil
    bitsize = 256
    assert 0 <= ceil(int(integer).bit_length()/8) <= 32
    byte_repr = int(integer).to_bytes(length=bitsize // 8, byteorder="big", signed=False)
    assert len(byte_repr) == num_bytes
    return byte_repr


def equation2list_coeffs(equation, input_variables, output_variables, max_degree, deg1_input_var=None, verbose=False):
    """
    // monomial ordering (example with small system)
    //  input_variables=[x,y],  output_variables=[z,t],  max_degree=3
    //  ordering: [1, x, y, x^2, x*y, y^2, x^3, ..., y^3 | z, z*x, z*y, z*x*y, z*x^2, ..., z*y^2 | t, t*x, t*y, t*x*y, t*x^2, ..., t*y^2]
    """

    assert max_degree == equation.degree(), f"{max_degree}!={equation.degree()}\n{equation}"

    coeff2str = functools.partial(int2binary, num_bytes=32)
    one = equation.base_ring().one()
    list_str_coeffs = []
    list_monomials = []
    num_zero_coeffs = 0

    cte_coeff = equation.constant_coefficient()
    list_str_coeffs.append(coeff2str(cte_coeff))
    list_monomials.append(one)
    num_zero_coeffs += int(cte_coeff == 0)
    if verbose:
        print(list_str_coeffs[-1], list_str_coeffs[-1])

    for out_var in [one] + list(output_variables):
        offset_d = 1 if out_var == one else 0
        for input_degree in range(offset_d, max_degree + offset_d):
            for in_mon in itertools.combinations_with_replacement(input_variables, input_degree):
                if deg1_input_var is not None and in_mon.count(deg1_input_var) >= 2:
                    continue
                in_mon = prod(in_mon, one)
                monomial = out_var * in_mon
                coeff = equation.monomial_coefficient(monomial)
                list_str_coeffs.append(coeff2str(coeff))
                list_monomials.append(monomial)
                num_zero_coeffs += int(coeff == 0)
                if verbose:
                    print(list_str_coeffs[-1], list_str_coeffs[-1])

    assert set(equation.monomials()).issubset(set(list_monomials))

    assert equation.number_of_terms() <= len(list_str_coeffs) == len(list_monomials)
    return list_str_coeffs, list_monomials, num_zero_coeffs


if __name__ == '__main__':
    list_round_equations_p, equations_n = generate_equations(True, info_filename)

    if FAST_MODE:
        X_in, Y_in, KA_in, M_in, Bit_in, X_out, Y_out, KA_out, M_out = PRp.gens()
        x_in, ka_in, m_in, r_out, s_out = PRn.gens()
    else:
        X_in, Y_in, KA_in, KB_in, M_in, Z_in, Bit_in, X_out, Y_out, KA_out, KB_out, M_out, Z_out = PRp.gens()
        # x_in, ka_in, kb_in, m_in, z_in, r_out, s_out = PRn.gens()
        warnings.warn("adding z_out")
        x_in, ka_in, kb_in, m_in, z_in, r_out, s_out, z_out = PRn.gens()

    # # example
    # print("equation 10")
    # print(list_round_equations_p[10][0])
    # print()
    # eq = list_round_equations_p[10][0]
    # input_variables = [X_in, Y_in, KA_in, KB_in, M_in, Z_in, Bit_in]
    # output_variables = [X_out, Y_out, KA_out, KB_out, M_out, Z_out]
    # max_degree = 3
    # list_str_coeffs, monomials, num_zero_coeffs = equation2list_coeffs(
    #     eq, input_variables, output_variables, max_degree, deg1_input_var=Bit_in)
    # print("monomials, num_zero_coeffs and list_str_coeffs of equation 10:")
    # print(monomials)
    # print(num_zero_coeffs)
    # print(list_str_coeffs)
    # exit(-1)

    ## EQUATIONS MOD P

    list_eq_coeffs = []
    list_num_zero_coeffs = []
    if FAST_MODE:
        input_variables_first_round = [X_in, Bit_in]
        input_variables_middlelast_rounds = [X_in, Y_in, KA_in, M_in, Bit_in]
        output_variables_firstmiddle_rounds = [X_out, Y_out, KA_out, M_out]
        output_variables_last_round = [X_out, KA_out, M_out]
    else:
        input_variables_first_round = [Z_in, Bit_in]
        input_variables_middlelast_rounds = [X_in, Y_in, KA_in, KB_in, M_in, Z_in, Bit_in]
        output_variables_firstmiddle_rounds = [X_out, Y_out, KA_out, KB_out, M_out, Z_out]
        output_variables_last_round = [X_out, KA_out, KB_out, M_out, Z_out]
    monomials_first_round = []
    monomials_middle_rounds = []
    monomials_last_round = []
    num_total_coeffs = 0
    for index_round, list_eq_p in enumerate(list_round_equations_p):
        max_degree = 4 if index_round == len(list_round_equations_p) - 1 else 3
        if index_round == 0:
            input_variables = input_variables_first_round
            output_variables = output_variables_firstmiddle_rounds
            monomials = monomials_first_round
        elif index_round < len(list_round_equations_p) - 1:
            input_variables = input_variables_middlelast_rounds
            output_variables = output_variables_firstmiddle_rounds
            monomials = monomials_middle_rounds
        else:
            assert index_round == len(list_round_equations_p) - 1
            input_variables = input_variables_middlelast_rounds
            output_variables = output_variables_last_round
            monomials = monomials_last_round
        for eq in list_eq_p:
            eq_coeffs, eq_monomials, eq_num_zero_coeffs = equation2list_coeffs(
                eq, input_variables, output_variables, max_degree, deg1_input_var=Bit_in)
            if len(monomials) == 0:
                monomials.extend(eq_monomials)
            assert monomials == eq_monomials
            list_eq_coeffs.append(eq_coeffs)
            list_num_zero_coeffs.append(eq_num_zero_coeffs)
            num_total_coeffs += len(eq_coeffs)

    num_equations_first_round = len(list_round_equations_p[0])
    num_equations_middle_rounds = len(list_round_equations_p[1])
    num_equations_last_round = len(list_round_equations_p[-1])
    assert all(num_equations_middle_rounds == len(list_eq_p) for list_eq_p in list_round_equations_p[1:-1])

    num_coeffs_pereq_first_round = len(list_eq_coeffs[0])
    num_coeffs_pereq_middle_rounds = len(list_eq_coeffs[num_equations_first_round])
    num_coeffs_pereq_last_round = len(list_eq_coeffs[-1])
    assert all(num_coeffs_pereq_first_round == len(eq_coeffs) for eq_coeffs in list_eq_coeffs[:num_equations_first_round])
    assert all(num_coeffs_pereq_middle_rounds == len(eq_coeffs) for eq_coeffs in list_eq_coeffs[num_equations_first_round:-num_equations_last_round])
    assert all(num_coeffs_pereq_last_round == len(eq_coeffs) for eq_coeffs in list_eq_coeffs[-num_equations_last_round:])

    smart_print_nonbyte = get_smart_print(filename, byte_mode=False)
    smart_print_byte = get_smart_print(filename, byte_mode=True, encoding_mode=ENCODING_MODE)
    smart_print_info = get_smart_print(info_filename, byte_mode=False)

    string_length = 32

    # equations degrees in smart_print_info hardcoded
    smart_print_info(f"// key: {hex(SECRET_KEY)}\n")
    smart_print_info(f"// EQUATIONS IN Fp:\n")
    smart_print_info(f"// number of rounds: {NUM_ROUNDS}\n")
    smart_print_info(f"// number of characters per Fp coefficient: {string_length}\n")
    smart_print_info(f"// number of zero coefficients in each Fp-equation: {list_num_zero_coeffs}\n")
    smart_print_info(f"// equations 1st round:\n")
    smart_print_info(f"//  - input variables: {input_variables_first_round} ({Bit_in} only appears with degree 0 and 1)\n")
    smart_print_info(f"//  - output variables: {output_variables_firstmiddle_rounds}\n")
    smart_print_info(f"//  - {num_equations_first_round} equations, each one with {num_coeffs_pereq_first_round} coeffs\n")
    smart_print_info(f"//  - equation degree: 3\n")
    smart_print_info(f"//  - monomial ordering used (num monomials={len(monomials_first_round)}): {monomials_first_round}\n")
    smart_print_info(f"// equations middle round (info for each round):\n")
    smart_print_info(f"//  - input variables: {input_variables_middlelast_rounds} ({Bit_in} only appears with degree 0 and 1)\n")
    smart_print_info(f"//  - output variables: {output_variables_firstmiddle_rounds}\n")
    smart_print_info(f"//  - equation degree: 3\n")
    smart_print_info(f"//  - {num_equations_middle_rounds} equations, each one with {num_coeffs_pereq_middle_rounds} coeffs\n")
    smart_print_info(f"//  - monomial ordering used (num monomials={len(monomials_middle_rounds)}): {monomials_middle_rounds}\n")
    smart_print_info(f"// equations last rounds:\n")
    smart_print_info(f"//  - input variables: {input_variables_middlelast_rounds} ({Bit_in} only appears with degree 0 and 1)\n")
    smart_print_info(f"//  - output variables: {output_variables_last_round}\n")
    smart_print_info(f"//  - equation degree: 4\n")
    smart_print_info(f"//  - {num_equations_last_round} equations, each one with {num_coeffs_pereq_last_round} coeffs\n")
    smart_print_info(f"//  - monomial ordering used (num monomials={len(monomials_last_round)}): {monomials_last_round}\n\n")

    number_of_strings = num_total_coeffs
    # smart_print_nonbyte(f'const unsigned char coeffs_p[{number_of_strings*string_length}] = "')
    # # using string literal instead of char array
    smart_print_nonbyte(f'const unsigned char* coeffs_p = "')
    for index_eq_coeffs, eq_coeffs in enumerate(list_eq_coeffs):
        for index_str_coeff, str_coeff in enumerate(eq_coeffs):
            assert string_length == len(str_coeff)
            smart_print_byte(str_coeff)
    smart_print_nonbyte('";\n')

    ## EQUATIONS MOD N

    if FAST_MODE:
        input_variables = [x_in, ka_in, m_in]
        max_degree = 2
        output_variables = [r_out, s_out]
    else:
        input_variables = [x_in, ka_in, kb_in, m_in, z_in]
        # max_degree = 5
        warnings.warn("changed degree")
        max_degree = 6
        warnings.warn("changed output vars")
        output_variables = [r_out, s_out, z_out]
    monomials = None
    list_eq_coeffs = []
    list_num_zero_coeffs = []
    for eq in equations_n:
        eq_coeffs, eq_monomials, eq_num_zero_coeffs = equation2list_coeffs(
            eq, input_variables, output_variables, max_degree)
        if monomials is None:
            monomials = eq_monomials
        assert monomials == eq_monomials
        list_eq_coeffs.append(eq_coeffs)
        list_num_zero_coeffs.append(eq_num_zero_coeffs)

    num_equations = len(equations_n)
    num_coeffs_per_equation = len(list_eq_coeffs[0])
    assert all(num_coeffs_per_equation == len(eq_coeffs) for eq_coeffs in list_eq_coeffs)
    num_total_coeffs = num_equations * num_coeffs_per_equation

    string_length = 32

    smart_print_info(f"\n\n// EQUATIONS IN Fn:\n")
    smart_print_info(f"// number of characters per Fn coefficient: {string_length}\n")
    smart_print_info(f"// {num_equations} equations, each one with {num_coeffs_per_equation} coeffs\n")
    smart_print_info(f"// number of zero coefficients in each Fn-equation: {list_num_zero_coeffs}\n")
    smart_print_info(f"// input variables: {input_variables}\n")
    smart_print_info(f"// output variables: {output_variables}\n")
    smart_print_info(f"// equation degree: {max_degree}\n")
    smart_print_info(f"// monomial ordering used (num monomials={len(monomials)}): {monomials}\n\n")

    number_of_strings = num_total_coeffs
    # smart_print_nonbyte(f'const unsigned char coeffs_n[{number_of_strings*string_length}] = "')
    # # using string literal instead of char array
    smart_print_nonbyte(f'const unsigned char* coeffs_n = "')
    for index_eq_coeffs, eq_coeffs in enumerate(list_eq_coeffs):
        for index_str_coeff, str_coeff in enumerate(eq_coeffs):
            assert string_length == len(str_coeff)
            smart_print_byte(str_coeff)
    smart_print_nonbyte('";\n')
