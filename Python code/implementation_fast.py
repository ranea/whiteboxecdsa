import collections
import itertools
import os
import contextlib

from sage.all import (
    Integers, GF, EllipticCurve, PolynomialRing, matrix, identity_matrix,
    ideal, random_matrix, vector, random_vector, set_random_seed, prod
)

from commands import NIST_P256, ecdsa_verify_str

from script_generate_overflows import get_list_overflows


DEBUG_MODE = False
DEBUG_OVERFLOW_MODE = False
if DEBUG_MODE:
    assert DEBUG_OVERFLOW_MODE
    print("WARNING: DEBUG_MODE ENABLED")
if DEBUG_OVERFLOW_MODE:
    print("WARNING: DEBUG_OVERFLOW_MODE ENABLED")


# secret key d (as integer) and public key Q = d G as hexadecimal string pair
SECRET_KEY = None  # e.g., 0x9C29EDDAEF2C2B4452052B668B83BE6365004278068884FA1AC3F6D0622875C3
Q_x_str = None  # e.g., "78E0E9DACCC47DE94D674DF3B35624A2F08E600B26B3444077022AD575AF4DB7"
Q_y_str = None  # e.g., "3084B4B8657EEA12396FDE260432BA7BDB3E092D61A42F830150D6CC8D798F9F"
PUBLIC_KEY_str = Q_x_str + Q_y_str
assert str(NIST_P256.scalar_multiplication(SECRET_KEY)) == PUBLIC_KEY_str


# NIST P-256 CONSTANTS
P = 2 ** 256 - 2 ** 224 + 2 ** 192 + 2 ** 96 - 1
A = P - 3
B = 41058363725152142129326129780047268409114441015993725554835256314039467401291
N = 115792089210356248762697446949407573529996955224135760342422259061068512044369
Gx = 48439561293906451759052585252797914202762949526041747995844080717082404635286
Gy = 36134250956749795798585127919587881956611106672985015071877198253568414405109

ZZ = Integers()
Fp = GF(P)
Fn = GF(N)

NISTP256 = EllipticCurve(Fp, [A, B])
NISTP256.set_order(115792089210356248762697446949407573529996955224135760342422259061068512044369)
G = NISTP256(Gx, Gy)


# OTHER CONSTANTS
seed = None  # e.g., 0
NUM_ROUNDS = 256
MAX_ROW_WEIGHT = 4


# POLYNOMIAL RINGS AND SYMBOLIC VARIABLES
PRp = PolynomialRing(Fp, names="X_in, Y_in, KA_in, M_in, Bit_in, X_out, Y_out, KA_out, M_out")
PRn = PolynomialRing(Fn, names="x_in, ka_in, m_in, r_out, s_out")
# X_in, Y_in, KA_in, M_in, Bit_in, X_out, Y_out, KA_out, M_out = PRp.gens()
# x_in, ka_in, m_in, r_out, s_out = PRn.gens()


# ----------


def get_smart_print(filename):
    if filename is None:
        def smart_print(*args):
            print(*args)
    else:
        def smart_print(*args):
            with open(filename, "a") as f:
                print(*args, file=f)
    return smart_print


if isinstance(seed, str) and not seed.isdigit():
    seed = abs(hash(seed))
set_random_seed(seed)


def sample_curve_fixed_points():
    # 1st - sample 1 list (containing the (ki0, ki1) pairs), verifying
    #       that the sum of of all max(ki0, ki1) is less than N - 1
    k_set = set()

    n_over_rounds = int(N // NUM_ROUNDS)

    def get_k_list():
        k_list = []
        max_sum_ki = 0
        for _ in range(NUM_ROUNDS - 1):
            while True:
                ki0 = Fp(ZZ.random_element(1, n_over_rounds + 1))
                ki1 = Fp(ZZ.random_element(1, n_over_rounds + 1))
                if ki0 != ki1 and ki0 not in k_set and ki1 not in k_set:
                    break
            k_list.append((ki0, ki1))
            k_set.add(ki0)
            k_set.add(ki1)
            max_sum_ki += max(ki0, ki1)

        while True:
            ki0 = Fp(ZZ.random_element(1, (N - 1) - max_sum_ki + 1))
            ki1 = Fp(ZZ.random_element(1, (N - 1) - max_sum_ki + 1))
            if ki0 != ki1 and ki0 not in k_set and ki1 not in k_set:
                break
        k_list.append((ki0, ki1))
        assert sum(ki0 for ki0, _ in k_list) <= N - 1
        assert sum(ki1 for _, ki1 in k_list) <= N - 1
        return k_list

    ka_list = get_k_list()

    k_list = []
    points = []
    for i, (ka_i0, ka_i1) in enumerate(ka_list):
        if DEBUG_MODE:
            ka_i0, ka_i1 = 2*i + 1, 2*i + 1
        k_list.append((ka_i0, ka_i1))
        points.append((ZZ(ka_i0) * G, ZZ(ka_i1) * G))

    return k_list, points


def get_small_Fp_square_matrix(num_rows):
    """
    small Fp matrices to be used to change from Fp to Fn
    all entries are non-zero and all rows have the same weight (which is minimized)
    """
    assert num_rows == 3
    if num_rows == 3:
        m = matrix(Fp, 3, 3, [[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    elif num_rows == 4:
        m = matrix(Fp, 4, 4, [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    else:
        assert num_rows == 5
        # row weight = 6
        # m = matrix(Fp, 5, 5, [[2, 1, 1, 1, 1], [1, 2, 1, 1, 1], [1, 1, 2, 1, 1], [1, 1, 1, 2, 1], [1, 1, 1, 1, 2]])

        # row weight = 7
        m = matrix(Fp, 5, 5, [[2, 1, 2, 1, 1], [2, 2, 1, 1, 1], [1, 1, 2, 1, 2], [1, 2, 1, 2, 1], [1, 1, 1, 2, 2]])
    for row in m.rows():
        assert sum(row) == MAX_ROW_WEIGHT
    assert m.is_invertible()
    return m


def sample_poly(poly_vars, finite_field, degree):
    assert degree in [0, 1, 2, 3, 4, 5]

    if 1 not in poly_vars:
        poly_vars = [1] + list(poly_vars)

    if finite_field == Fp:
        Bit_in = PRp("Bit_in")
    else:
        Bit_in = None

    if DEBUG_MODE:
        return 1

    poly = finite_field.random_element()
    if degree == 0:
        return poly
    for var in poly_vars:
        poly += finite_field.random_element() * var
        if degree in [2, 3, 4, 5]:
            for other_var in poly_vars:
                if var == other_var == Bit_in:
                    continue
                poly += finite_field.random_element() * var * other_var
                if degree in [3, 4, 5]:
                    for other_other_var in poly_vars:
                        if Bit_in in [var, other_var] and Bit_in == other_other_var:
                            continue
                        poly += finite_field.random_element() * var * other_var * other_other_var
                        if degree in [4, 5]:
                            for other_other_other_var in poly_vars:
                                if Bit_in in [var, other_var, other_other_var] and Bit_in == other_other_other_var:
                                    continue
                                poly += finite_field.random_element() * var * other_var * other_other_var * other_other_other_var
                                if degree in [5]:
                                    for other_other_other_other_var in poly_vars:
                                        if Bit_in in [var, other_var, other_other_var, other_other_other_var] and Bit_in == other_other_other_other_var:
                                            continue
                                        poly += finite_field.random_element() * var * other_var * other_other_var * other_other_other_var * other_other_other_other_var
    return poly


def sample_poly_with_output_vars(input_vars, output_vars, finite_field, max_degree):
    one = finite_field.one()
    poly = finite_field.random_element()  # cte coeff

    if DEBUG_MODE:
        return 1

    if max_degree == 0:
        return poly
    for out_var in [one] + list(output_vars):
        offset_d = 1 if out_var == one else 0
        for input_degree in range(offset_d, max_degree + offset_d):
            for in_mon in itertools.combinations_with_replacement(input_vars, input_degree):
                in_mon = prod(in_mon, one)
                monomial = out_var * in_mon
                coeff = finite_field.random_element()
                poly += coeff * monomial
    return poly


def count_zero_coeffs(equation, input_vars, output_vars, finite_field, max_degree):
    one = finite_field.one()
    num_zero_coeffs = 0
    for out_var in [one] + list(output_vars):
        offset_d = 1 if out_var == one else 0
        for input_degree in range(offset_d, max_degree + offset_d):
            for in_mon in itertools.combinations_with_replacement(input_vars, input_degree):
                in_mon = prod(in_mon, one)
                monomial = out_var * in_mon
                coeff = equation.monomial_coefficient(monomial)
                num_zero_coeffs += int(coeff == 0)
    return num_zero_coeffs


def remove_Bit_in_powers(eqn):
    """Reduce equation by replacing Bit_in^2 by Bit_in (since Bit_in only takes 0 or 1)."""
    Bit_in = PRp("Bit_in")
    zero = eqn.subs({Bit_in: 0})
    one = eqn.subs({Bit_in: 1})
    return zero + (one - zero) * Bit_in


def generate_equations(verbose=False, filename=None):
    k_list, points = sample_curve_fixed_points()

    X_in, Y_in, KA_in, M_in, Bit_in, X_out, Y_out, KA_out, M_out = PRp.gens()
    x_in, ka_in, m_in, r_out, s_out = PRn.gens()

    round_encodings = [[None, None] for _ in range(NUM_ROUNDS + 1)]  # input/output encoding of each round

    # - encodings all round except last

    round_encodings[0][0] = {}  # input encoding of 1st round is the identity

    for round in range(NUM_ROUNDS - 1):  # except last Fp-round
        output_vars = vector(PRp, [X_out, Y_out, KA_out, M_out])
        input_vars_next_round = vector(PRp, [X_in, Y_in, KA_in, M_in])
        while True:
            A = random_matrix(Fp, len(output_vars), len(output_vars))
            b = random_vector(Fp, len(output_vars))
            if A.is_invertible() and 0 not in A.list() and 0 not in b.list():
                break
        if DEBUG_MODE:
            A = identity_matrix(Fp, len(output_vars), len(output_vars))
            b = vector(Fp, [0 for _ in range(len(output_vars))])
        output_encoding = A * output_vars + b
        output_replacement = {output_vars[i]: output_encoding[i] for i in range(len(output_vars))}
        input_encoding = A * input_vars_next_round + b
        input_replacement = {input_vars_next_round[i]: input_encoding[i] for i in range(len(input_vars_next_round))}
        round_encodings[round][1] = output_replacement
        round_encodings[round + 1][0] = input_replacement  # no need to invert

    output_vars = vector(PRp, [X_out, KA_out, M_out])  # no need Y_out
    small_matrix = get_small_Fp_square_matrix(len(output_vars))
    while True:
        # b = vector(ZZ, [-int(ZZ.random_element(N)) for _ in range(len(output_vars))])
        b = vector(ZZ, [int(ZZ.random_element(N // MAX_ROW_WEIGHT)) for _ in range(len(output_vars))])
        if 0 not in b.list():
            break
    if DEBUG_MODE or DEBUG_OVERFLOW_MODE:
        small_matrix = identity_matrix(Fp, len(output_vars), len(output_vars))
        b = vector(ZZ, [0 for _ in range(len(output_vars))])

    assert matrix(Fp, small_matrix).change_ring(ZZ) == matrix(Fn, small_matrix).change_ring(ZZ)
    assert (matrix(Fp, small_matrix) * vector(Fp, list(b))).change_ring(ZZ) == \
           (matrix(Fn, small_matrix) * vector(Fn, list(b))).change_ring(ZZ)
    assert ((matrix(Fp, small_matrix) ** (-1)) * (matrix(Fp, small_matrix) * vector(Fp, list(b)))).change_ring(ZZ) == \
           ((matrix(Fn, small_matrix) ** (-1)) * (matrix(Fn, small_matrix) * vector(Fn, list(b)))).change_ring(ZZ)

    # A(x) = L(x) + b
    # A^(-1)(x) = L^{-1}(x) + L^{-1}(b)
    inv_cte = (matrix(Fp, small_matrix) ** (-1)) * vector(Fp, list(b))
    output_encoding = (matrix(Fp, small_matrix) ** (-1)) * output_vars + inv_cte
    output_replacement = {output_vars[i]: output_encoding[i] for i in range(len(output_vars))}

    input_vars_next_round = vector(PRn, [x_in, ka_in, m_in])
    inv_cte = (matrix(Fn, small_matrix) ** (-1)) * vector(Fn, list(b))
    input_encoding = (matrix(Fn, small_matrix) ** (-1)) * input_vars_next_round + inv_cte
    input_replacement = {input_vars_next_round[i]: input_encoding[i] for i in range(len(input_vars_next_round))}

    round_encodings[NUM_ROUNDS - 1][1] = output_replacement
    round_encodings[NUM_ROUNDS][0] = input_replacement
    round_encodings[NUM_ROUNDS][1] = {}  # output encoding of Fn-round is the identity

    #  - generate equations

    list_round_equations_p = []

    for round in range(NUM_ROUNDS):
        if round == 0:
            KA_in, M_in = 0, 0
            input_vars = [X_in, Bit_in]  # x_in only used to multiply equations
        else:
            KA_in, M_in = PRp("KA_in"), PRp("M_in")
            input_vars = [X_in, Y_in, KA_in, M_in, Bit_in]

        # (point_x, point_y) = (1-m_i)(k_i0 * G) + m_i(k_i1 * G)
        point_x = (1 - Bit_in) * points[round][0][0] + Bit_in * points[round][1][0]
        point_y = (1 - Bit_in) * points[round][0][1] + Bit_in * points[round][1][1]

        if round == 0:
            eq1 = X_out - point_x
            eq1 *= sample_poly(input_vars, Fp, 2)
        else:
            eq1 = (point_y - Y_in) ** 2 - (X_in + point_x + X_out) * (point_x - X_in) ** 2
            if round == NUM_ROUNDS - 1:
                eq1 *= sample_poly(input_vars, Fp, 1)

        if round < NUM_ROUNDS - 1:
            if round == 0:
                eq2 = Y_out - point_y
                eq2 *= sample_poly(input_vars, Fp, 2)
            else:
                eq2 = ((point_y - Y_in) * (X_in - X_out) - (Y_out + Y_in) * (point_x - X_in))
                eq2 *= sample_poly(input_vars, Fp, 1 + int(round == NUM_ROUNDS - 1))
        else:
            eq2 = None

        ka_i0_val, ka_i1_val = k_list[round]

        KA_update = (1 - Bit_in) * ka_i0_val + Bit_in * ka_i1_val
        assert DEBUG_MODE or KA_update.degree() == 1, str(KA_update.degree())
        eq6 = (KA_out - KA_in - KA_update)
        eq6 *= sample_poly(input_vars, Fp, 2 + int(round == NUM_ROUNDS - 1))

        M_update = Bit_in * (2 ** round)
        assert DEBUG_MODE or M_update.degree() == 1
        eq4 = (M_out - M_in - M_update)
        eq4 *= sample_poly(input_vars, Fp, 2 + int(round == NUM_ROUNDS - 1))

        if not DEBUG_MODE:
            d = 3 if round < NUM_ROUNDS - 1 else 4
            assert eq1.degree() == d, str(eq1.degree())
            if round < NUM_ROUNDS - 1:
                assert eq2.degree() == d, str(eq2.degree())
            assert eq6.degree() == d, str(eq6.degree())
            assert eq4.degree() == d, str(eq4.degree())

        if round < NUM_ROUNDS - 1:
            round_equations = [eq1, eq2, eq6, eq4]
        else:
            round_equations = [eq1, eq6, eq4]

        # apply input and output encodings and reduce Bit_in
        aux = []
        for eq in round_equations:
            eq = eq.subs(round_encodings[round][0])
            eq = eq.subs(round_encodings[round][1])
            eq = remove_Bit_in_powers(eq)
            aux.append(eq)
        round_equations = aux

        while True:
            left_matrix = random_matrix(Fp, len(round_equations), len(round_equations))
            if left_matrix.is_invertible() and 0 not in left_matrix.list():
                break
        if DEBUG_MODE:
            left_matrix = identity_matrix(Fp, len(round_equations), len(round_equations))
        list_round_equations_p.append(left_matrix * vector(PRp, round_equations))

    input_vars = [x_in, ka_in, m_in]
    eq1 = s_out * (ka_in) - (m_in + x_in * SECRET_KEY)
    eq2 = (r_out - x_in)
    assert DEBUG_MODE or eq1.degree() == 2, str(eq1.degree())
    assert DEBUG_MODE or eq2.degree() == 1, str(eq2.degree())

    # eq1 *= sample_poly(input_vars, Fn, 0)
    eq2 *= sample_poly(input_vars, Fn, 1)
    assert DEBUG_MODE or eq1.degree() == 2, str(eq1.degree())
    assert DEBUG_MODE or eq2.degree() == 2, str(eq2.degree())

    equations_n = [eq1, eq2]

    equations_n = [eq.subs(round_encodings[NUM_ROUNDS][0]) for eq in equations_n]  # input encoding
    assert round_encodings[NUM_ROUNDS][1] == {}  # output encoding

    while True:
        left_matrix = random_matrix(Fn, len(equations_n), len(equations_n))
        if left_matrix.is_invertible() and 0 not in left_matrix.list():
            break
    if DEBUG_MODE:
        left_matrix = identity_matrix(Fn, len(equations_n), len(equations_n))
    equations_n = left_matrix * vector(PRn, equations_n)

    # for eq in equations_n:
    #     print(count_zero_coeffs(eq, input_vars, [r_out, s_out], Fn, eq.degree()))
    # exit(-1)

    coeffs = sum([len(eq.coefficients()) for list_eq in list_round_equations_p + [equations_n] for eq in list_eq])
    if verbose:
        smart_print = get_smart_print(filename)
        smart_print("number of coefficients = %d" % coeffs)
        smart_print("size = %d MB" % (coeffs * 32 * 1e-6))
        smart_print()

    return list_round_equations_p, equations_n


def verify(message, r, s, curve_points=None, verbose=False, filename=None):
    # # ECDSA verification (very slow)
    r_hex = hex(int(r))[2:]
    r_hex = "0" * (64 - len(r_hex)) + r_hex
    s_hex = hex(int(s))[2:]
    s_hex = "0" * (64 - len(s_hex)) + s_hex
    return ecdsa_verify_str(PUBLIC_KEY_str, hex(message), r_hex + s_hex, verbose=verbose, filename=filename)

    # # # plain evaluation
    # message_bits = bin(message)[2:]
    # message_bits = "0" * (256 - len(message_bits)) + message_bits
    # assert len(message_bits) == 256
    # message_bits = [int(bit) for bit in reversed(message_bits)]
    #
    # k_0, P_0, ki, points = curve_points
    # m = 0
    # k = Fn(k_0)
    # Pt = P_0
    #
    # for i in range(NUM_ROUNDS):
    #     m = m + message_bits[i] * (2 ** i)
    #     k = k + Fn(ki[i][message_bits[i]])
    #     Pt = Pt + points[i][message_bits[i]]
    #
    # assert m == message, f"{m} != {message}"
    # x_plain = Fn(Pt[0])
    # s_plain = Fn(k) ** (-1) * (Fn(m) + Fn(Pt[0]) * SECRET_KEY)
    #
    # return r == x_plain and s == s_plain


def get_message_bits(message):
    message_bits = bin(message)[2:]
    message_bits = "0" * (256 - len(message_bits)) + message_bits
    assert len(message_bits) == 256
    return [int(bit) for bit in reversed(message_bits)]


def check_correctness(number_iterations=1, print_intermediate_values=False, filename=None):
    smart_print = get_smart_print(filename)

    list_round_equations_p, equations_n = generate_equations(print_intermediate_values, filename)  # equations.c

    # in C source use list_overflows.c
    list_overflows = get_list_overflows(3, (MAX_ROW_WEIGHT - 1) + 1, verbose=False)

    if DEBUG_MODE or DEBUG_OVERFLOW_MODE:
        list_overflows = list(itertools.product([0], repeat=3))

    X_in, Y_in, KA_in, M_in, Bit_in, X_out, Y_out, KA_out, M_out = PRp.gens()
    x_in, ka_in, m_in, r_out, s_out = PRn.gens()

    overflow_counter = collections.Counter()

    for index_iteration in range(number_iterations):
        message = int(ZZ.random_element(2**256))

        message_bits = get_message_bits(message)
        assert len(message_bits) == NUM_ROUNDS == 256

        smart_print(f"- white-box evaluation {index_iteration + 1} out of {number_iterations}:")
        smart_print("message:", message)
        smart_print("message bits (from LSB to MSB):", message_bits, "\n")

        z = Fp(message)

        while True:  # implement in C as LABEL+GOTO (see below)
            if print_intermediate_values:
                smart_print(f"\tinput variables round {0}")
                smart_print(f"\t - z: {z}\n\t - Bit_in: {message_bits[0]}")

            for i in range(NUM_ROUNDS):
                round_equations_p = list_round_equations_p[i]

                # substitute inputs
                if i == 0:
                    substitutions = {X_in: z, Bit_in: message_bits[i]}
                else:
                    substitutions = {X_in: x, Y_in: y, KA_in: ka, M_in: m, Bit_in: message_bits[i]}
                round_equations_p = [eq.subs(substitutions) for eq in round_equations_p]

                # solve equations (in the C file, groebner is replaced by Gauss)
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    GB = ideal(round_equations_p).groebner_basis()

                # we assume GB is of the form X-x, ..., K-k, M-m, ...
                if i < NUM_ROUNDS - 1:
                    x =  -GB[0](0, 0, 0, 0, 0, 0, 0, 0, 0)
                    y =  -GB[1](0, 0, 0, 0, 0, 0, 0, 0, 0)
                    ka = -GB[2](0, 0, 0, 0, 0, 0, 0, 0, 0)
                    m =  -GB[3](0, 0, 0, 0, 0, 0, 0, 0, 0)
                else:
                    x =  -GB[0](0, 0, 0, 0, 0, 0, 0, 0, 0)
                    ka = -GB[1](0, 0, 0, 0, 0, 0, 0, 0, 0)
                    m =  -GB[2](0, 0, 0, 0, 0, 0, 0, 0, 0)

                if print_intermediate_values:
                    smart_print(f"\toutput variables obtained in round {i} (input Bit_in: {message_bits[i]})")
                    smart_print(f"\t - x: {x}\n\t - y: {y if i < NUM_ROUNDS - 1 else None}\n"
                                f"\t - ka: {ka}\n\t - m: {m}")

                if DEBUG_MODE:
                    kG = NIST_P256.scalar_multiplication(ka)
                    assert kG.x.val == x, f"{kG.x.val} != {x}"
                    if i < NUM_ROUNDS - 1:
                        assert kG.y.val == y, f"{kG.y.val} != {y}"

            if print_intermediate_values:
                smart_print("")

            if DEBUG_MODE:
                kG = NIST_P256.scalar_multiplication(ka)
                assert kG.x.val == x, f"{kG.x.val} != {x}"

            found_overflow = False
            for overflows in list_overflows:
                # substitute inputs
                substitutions = {
                    x_in: ZZ(x) + overflows[0] * P,
                    ka_in: ZZ(ka) + overflows[1] * P,
                    m_in: ZZ(m) + overflows[2] * P,
                 }
                sub_equations_n = [eq.subs(substitutions) for eq in equations_n]

                # solve equations (in the C file, groebner is replaced by Gauss)
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    GB = ideal(sub_equations_n).groebner_basis()

                r = -GB[0](0, 0, 0, 0, 0)
                s = -GB[1](0, 0, 0, 0, 0)

                if print_intermediate_values:
                    smart_print(f"\t(r, s) obtained with (x,k,m)-overflow {overflows}")
                    smart_print(f"\t - r: {r}\n\t - s: {s}")

                if verify(message, r, s, None, print_intermediate_values, filename):
                    found_overflow = True
                    overflow_counter[tuple([int(x) for x in overflows])] += 1
                    break

            if found_overflow:
                break
            else:
                raise ValueError("no signature found")  # ignore this in C file and replaced by goto
                # continue  # implemented in C by goto

        if print_intermediate_values:
            smart_print("")

        smart_print("valid (x,ka,m)-overflow:", str(overflows))
        smart_print(f"valid signature of message {message}:")
        smart_print("> r:", r)
        smart_print("> s:", s)
        smart_print("counter overflows:", overflow_counter)
        smart_print()


if __name__ == '__main__':
    check_correctness(number_iterations=32, print_intermediate_values=True, filename=None)

    # check_correctness(number_iterations=1, print_intermediate_values=True, filename="intermediate_values_fast.txt")
