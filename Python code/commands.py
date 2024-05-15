# Source https://github.com/CryptoExperts/whibox_contest_submission_server
#!/usr/bin/env python3

import click
import hashlib
import random

from nist_p256 import NIST_P256


@click.command()
@click.argument('seed', default="CHES2021")
def cmd_keygen(seed):
    # CHES 2021 will start from September 12, 2021
    random.seed(seed)
    d = random.randint(1, NIST_P256.n-1)
    Q = NIST_P256.scalar_multiplication(d)

    print(f"seed: seed = {seed}")
    print(f"private key: d = {d:064X}")
    print(f"public key:  Q = ({repr(Q)})")
    print(f"encoded public key:  {Q}")

    return d, Q


def aux_keygen(seed):
    """

        >>> import secrets
        >>> seed = secrets.randbits(128)
        >>> seed = 58884332834572034407025533416641542861
        >>> aux_keygen(seed)
        seed: seed = 58884332834572034407025533416641542861
        private key: d = 6F1D9093F3D5AE7C5F133659295914C9AF22E54B4ADE38CA421CA9BBD3D48A50
        public key:  Q = (x = 24A5B8DE827036C915D1A96D04C512EC548E45B36B48E16ADA083708512E2F69, y = 0FBA517095CD982552F6EDB78F3D4EF2800A767BD9469D3FA9E53E4AE2CF8537)
        encoded public key:  24A5B8DE827036C915D1A96D04C512EC548E45B36B48E16ADA083708512E2F690FBA517095CD982552F6EDB78F3D4EF2800A767BD9469D3FA9E53E4AE2CF8537
        (50258962597886030548374638065802159410786697345394157866664706117240104454736, x = 24A5B8DE827036C915D1A96D04C512EC548E45B36B48E16ADA083708512E2F69, y = 0FBA517095CD982552F6EDB78F3D4EF2800A767BD9469D3FA9E53E4AE2CF8537)
        >>> seed = 224678001425874045646222192685562089211
        >>> aux_keygen(seed)
        seed: seed = 224678001425874045646222192685562089211
        private key: d = ADA6C6A1049825989811C9495D83681A68C67AB5E8EBDDC126CEE77056A7BB27
        public key:  Q = (x = C14CD88E3676B4B5D79E6547ED90DA43F2381EC20CA251925B2D0A7A1A594743, y = 948AF7B0B5FD6AF4B97CCE4DF4F4848A536E6243EFC80ACC4922C0FD8900F1B7)
        encoded public key:  C14CD88E3676B4B5D79E6547ED90DA43F2381EC20CA251925B2D0A7A1A594743948AF7B0B5FD6AF4B97CCE4DF4F4848A536E6243EFC80ACC4922C0FD8900F1B7
        (78544790304470213684587094801034840001612314062934914751841603239807187139367, x = C14CD88E3676B4B5D79E6547ED90DA43F2381EC20CA251925B2D0A7A1A594743, y = 948AF7B0B5FD6AF4B97CCE4DF4F4848A536E6243EFC80ACC4922C0FD8900F1B7)

    """
    random.seed(seed)
    d = random.randint(1, NIST_P256.n-1)
    Q = NIST_P256.scalar_multiplication(d)

    print(f"seed: seed = {seed}")
    print(f"private key: d = {d:064X}")
    print(f"public key:  Q = ({repr(Q)})")
    print(f"encoded public key:  {Q}")

    return d, Q


@click.command()
@click.argument('pa_str', metavar="PUBLIC_KEY")
@click.argument('hash_', metavar="HASH")
@click.argument('signature', metavar="SIGNATURE")
def cmd_ecdsa_verify(pa_str: str, hash_: str, signature: str):
    if ecdsa_verify_str(pa_str, hash_, signature):
        print("Good signature :)")
    else:
        print("Wrong signature")
    return True


@click.command()
@click.argument('d_str', metavar="PRIVATE_KEY")
def cmd_ec_schnorr_sign(d_str: str):
    """Variables names follow BSI EC-Schnorr standardized"""
    d = decode_private(d_str)

    while True:
        # choose a random k
        k = random.randint(1, NIST_P256.n-1)

        # Q = k x G, r = Q[x]
        Q = NIST_P256.scalar_multiplication(k)
        Q_x = Q.x.val

        # h = SHA256(r)
        m = hashlib.sha256()
        m.update(bytes.fromhex(f"{Q_x:064x}"))
        r = int(m.hexdigest(), 16)
        if (r % NIST_P256.n) == 0:
            continue

        s = (k - r * d) % NIST_P256.n
        if s == 0:
            continue

        print("Signature:", f"{r:064X}{s:064X}")

        return r, s


@click.command()
@click.argument('pa_str', metavar="PUBLIC_KEY")
@click.argument('signature', metavar="SIGNATURE")
def cmd_ec_schnorr_verify(pa_str, signature):
    return ec_schnorr_verify(pa_str, signature)


def aux_ec_schnorr_sign(d_str: str, seed):
    """

        >>> seed = 137634784928364372021177876488069289618
        >>> private = "6F1D9093F3D5AE7C5F133659295914C9AF22E54B4ADE38CA421CA9BBD3D48A50"
        >>> public = "24A5B8DE827036C915D1A96D04C512EC548E45B36B48E16ADA083708512E2F690FBA517095CD982552F6EDB78F3D4EF2800A767BD9469D3FA9E53E4AE2CF8537"
        >>> aux_ec_schnorr_sign(private, seed)
        Signature: 8FC706F15F110EE8280C6F0D621133DBF7D973546DBEA1EDA82C96767D3ACB5CB9A55742EDE1B0557D1AA74AF58C2B3F6CFAD795BBFFA4D6733F48E0B1E6F0A8
        (65032387831134904602943104398085308184312058638075237643399354672836909910876, 83970009009933222188952299150680999928080735930252703627932853109968335204520)
        >>> ec_schnorr_verify(public, "8FC706F15F110EE8280C6F0D621133DBF7D973546DBEA1EDA82C96767D3ACB5CB9A55742EDE1B0557D1AA74AF58C2B3F6CFAD795BBFFA4D6733F48E0B1E6F0A8")
        Good signature :)
        True
        >>> seed = 36022124873950846300145156589815535577
        >>> private = "ADA6C6A1049825989811C9495D83681A68C67AB5E8EBDDC126CEE77056A7BB27"
        >>> public = "C14CD88E3676B4B5D79E6547ED90DA43F2381EC20CA251925B2D0A7A1A594743948AF7B0B5FD6AF4B97CCE4DF4F4848A536E6243EFC80ACC4922C0FD8900F1B7"
        >>> aux_ec_schnorr_sign(private, seed)
        Signature: B45DADFB1236F81CCA96EA62811222410ABE117224865C2988485E11074407B7C88344517D7D888B04A8D555934C45D7609694DECDD39961FEB37F1FCCEE5A86
        (81581830292995182135114581066067027608802972269294777338793963035718618187703, 90694498197862185711570960145169154206226223997478764926208343054661293333126)
        >>> ec_schnorr_verify(public, "B45DADFB1236F81CCA96EA62811222410ABE117224865C2988485E11074407B7C88344517D7D888B04A8D555934C45D7609694DECDD39961FEB37F1FCCEE5A86")
        Good signature :)
        True

    """
    random.seed(seed)
    d = decode_private(d_str)

    while True:
        # choose a random k
        k = random.randint(1, NIST_P256.n-1)

        # Q = k x G, r = Q[x]
        Q = NIST_P256.scalar_multiplication(k)
        Q_x = Q.x.val

        # h = SHA256(r)
        m = hashlib.sha256()
        m.update(bytes.fromhex(f"{Q_x:064x}"))
        r = int(m.hexdigest(), 16)
        if (r % NIST_P256.n) == 0:
            continue

        s = (k - r * d) % NIST_P256.n
        if s == 0:
            continue

        print("Signature:", f"{r:064X}{s:064X}")

        return r, s


def ec_schnorr_verify(pa_str, signature):
    """Variables names follow BSI EC-Schnorr standardized"""
    P_A = decode_public(pa_str)
    if not check_public_key(P_A):
        return False

    r, s = decode_signature(signature)
    if not check_r_s(r, s):
        return False

    Q = NIST_P256.scalar_multiplication(s) + \
        NIST_P256.scalar_multiplication(r, P_A)
    if Q.is_at_infinity:
        print("Wrong signature")
        return False

    m = hashlib.sha256()
    m.update(bytes.fromhex(f"{Q.x.val:064x}"))
    v = int(m.hexdigest(), 16)

    print("Good signature :)" if r == v else "Wrong signature")
    return r == v


def decode_signature(signature):
    if len(signature) != 128:
        raise click.ClickException(
            "SIGNATURE should be 128 hexadecimal digits long.")
    try:
        r = int(signature[:32*2], 16)
        s = int(signature[32*2:], 16)
    except ValueError:
        raise click.ClickException(
            "PUBLIC_KEY is not in valid hex.")
    return r, s


def decode_private(d_str):
    try:
        d = int(d_str, 16)
    except ValueError:
        raise click.ClickException("PRIVATE_KEY is not in valid hex.")
    return d


def decode_public(pa_str):
    if len(pa_str) != 128:
        raise click.ClickException(
            "PUBLIC_KEY should be 128 hexadecimal digits long.")
    try:
        pa_x = int(pa_str[:64], 16)
        pa_y = int(pa_str[64:], 16)
    except ValueError:
        raise click.ClickException("PUBLIC_KEY is not in valid hex.")
    return NIST_P256.Point(NIST_P256.Modular(pa_x), NIST_P256.Modular(pa_y))


def check_public_key(Q: NIST_P256.Point):
    if Q.is_at_infinity:
        print("Public key should not be infinity")
        return False

    if not Q.is_on_curve:
        print("Public key is not on curve")
        return False

    point_infinity = NIST_P256.scalar_multiplication(NIST_P256.n, Q)
    if not point_infinity.is_at_infinity:
        print("Something wrong with the public key")
        return False

    return True


def validate_private_key(d_str, pa_str):
    try:
        d = decode_private(d_str)
        pa = decode_public(pa_str)
        return pa == NIST_P256.scalar_multiplication(d)
    except:
        return False


def check_r_s(r, s):
    n = NIST_P256.n

    if r < 1 or r > n-1:
        print("r is not between [1, n-1]")
        return False

    if s < 1 or s > n-1:
        print("s is not between [1, n-1]")
        return False

    return True


def ecdsa_verify_str(pa_str: str, hash_: str, signature: str, verbose=False, filename=None):
    Q = decode_public(pa_str)
    hash_ = int(hash_, 16)
    r, s = decode_signature(signature)
    return ecdsa_verify(Q, hash_, (r, s), verbose, filename)


def ecdsa_verify(Q: NIST_P256.Point, hash_: int, signature: (int, int), verbose=False, filename=None):
    if not check_public_key(Q):
        return False

    r, s = signature
    if not check_r_s(r, s):
        return False

    from implementation_fast import get_smart_print
    smart_print = get_smart_print(filename)

    n = NIST_P256.n
    z = hash_ % n

    s_inv = int(pow(s, n-2, n))
    assert (s*s_inv) % n == 1
    u1 = (z * s_inv) % n
    u2 = (r * s_inv) % n

    P = NIST_P256.scalar_multiplication(u1) + \
        NIST_P256.scalar_multiplication(u2, Q)
    if P.is_at_infinity:
        print("Invalid signature")
        return False

    if verbose:
        smart_print(f"\t\tvefification of (r, s) = ({r}, {s}) of message {hash_}")
        smart_print(f"\t\t - s_inv: {s_inv}")
        smart_print(f"\t\t - u1, u2: {u1}, {u2}")
        smart_print(f"\t\t - P = u1 · G + u2 · Q: ({P.x.val}, {P.y.val}) ")
        smart_print(f"\t\t - (P.x mod n) == r: {(P.x.val % n) == r}")

    return (P.x.val % n) == r
