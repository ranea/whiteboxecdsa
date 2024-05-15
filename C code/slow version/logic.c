#include <gmp.h>
typedef char bool;
#define TRUE 1
#define FALSE 0
#define mat_size 6
typedef struct point {
    mpz_t x, y;
} __Point;
typedef __Point Point[1];
mpz_t n, p, a;
mpz_t Q_y_neg, lambda, denominator;
mpz_t inv, scale, inv_s, scale, scalar;
mpz_t mnml_base[253];
Point tmp, G, pk, PP, V1, V2;
void point_init(Point P) { mpz_inits(P->x, P->y, NULL); }
void point_init_set_str(Point P,
                        const char *x_str,
                        const char *y_str,
                        int base) {
    mpz_init_set_str(P->x, x_str, base);
    mpz_init_set_str(P->y, y_str, base);
}
void point_init_infinity(Point P) {
    mpz_init_set_ui(P->x, 0);
    mpz_init_set_ui(P->y, 0);
}
int point_is_infinity(const Point P) {
    return (mpz_cmp_ui(P->x, 0) == 0) && (mpz_cmp_ui(P->y, 0) == 0);
}
int point_equal(const Point P, const Point Q) {
    return (mpz_cmp(P->x, Q->x) == 0) && (mpz_cmp(P->y, Q->y) == 0);
}
int point_is_inverse(const Point P, const Point Q) {
    int comp = mpz_cmp(P->x, Q->x) == 0;
    if (comp != 1) {
        return comp;
    }
    mpz_neg(Q_y_neg, Q->y);
    comp = mpz_cmp(P->y, Q_y_neg) == 0;
    return comp;
}
void point_set(Point R, const Point P) {
    mpz_set(R->x, P->x);
    mpz_set(R->y, P->y);
}
void point_add(Point R, const Point P, const Point Q) {
    if (point_is_infinity(P)) {
        point_set(R, Q);
        return;
    } else if (point_is_infinity(Q)) {
        point_set(R, P);
        return;
    }
    if (point_is_inverse(P, Q)) {
        point_init_infinity(R);
        return;
    }
    if (P == Q || point_equal(P, Q)) {
        mpz_powm_ui(lambda, P->x, 2, p);
        mpz_mul_ui(lambda, lambda, 3);
        mpz_add(lambda, lambda, a);
        mpz_mul_ui(denominator, P->y, 2);
        mpz_invert(denominator, denominator, p);
    } else {
        mpz_sub(lambda, Q->y, P->y);
        mpz_sub(denominator, Q->x, P->x);
        mpz_invert(denominator, denominator, p);
    }
    mpz_mul(lambda, lambda, denominator);
    mpz_mod(lambda, lambda, p);
    mpz_powm_ui(R->x, lambda, 2, p);
    mpz_sub(R->x, R->x, P->x);
    mpz_sub(R->x, R->x, Q->x);
    mpz_mod(R->x, R->x, p);
    mpz_sub(R->y, P->x, R->x);
    mpz_mul(R->y, lambda, R->y);
    mpz_mod(R->y, R->y, p);
    mpz_sub(R->y, R->y, P->y);
    mpz_mod(R->y, R->y, p);
}
void point_scalar(Point R, const Point P, const mpz_t scalar) {
    const mp_bitcnt_t num_bits = 256;
    for (mp_bitcnt_t i = num_bits - 1; i >= 0 && i < num_bits; i--) {
        point_add(tmp, R, R);
        if (mpz_tstbit(scalar, i) == 1) {
            point_add(R, tmp, P);
        } else {
            point_set(R, tmp);
        }
    }
}
void GFLinearSolver(mpz_t coeff[mat_size][mat_size + 1], const mpz_t p){
    mpz_invert(inv, coeff[mat_size - 1][mat_size - 1], p);
    mpz_mul(coeff[mat_size - 1][mat_size], coeff[mat_size - 1][mat_size], inv);
    mpz_mod(coeff[mat_size - 1][mat_size], coeff[mat_size - 1][mat_size], p);
    for (int i = mat_size - 2; i >= 0; --i){
        for (int j = i + 1; j < mat_size; ++j)
            mpz_submul(coeff[i][mat_size], coeff[i][j], coeff[j][mat_size]);
        mpz_invert(inv, coeff[i][i], p);
        mpz_mul(coeff[i][mat_size], coeff[i][mat_size], inv);
        mpz_mod(coeff[i][mat_size], coeff[i][mat_size], p);
    }
}
void GaussElim(mpz_t coeff[mat_size][mat_size + 1], const mpz_t p){
    int target = 0;
    for (int k = 0; k < mat_size - 1; ++k){
        for (int i = k + 1; i < mat_size; ++i){
            target = k;
            while (target < mat_size){
                if (mpz_cmp_ui(coeff[target][k], 0))
                    break;
                target++;
            }
            if (target == mat_size)
                break;
            if (target != k)
                for (int i = 0; i < mat_size+1; ++i)
                    mpz_swap(coeff[k][i], coeff[target][i]);
            mpz_invert(inv, coeff[k][k], p);
            mpz_mul(scale, coeff[i][k], inv);
            mpz_mod(scale, scale, p);
            for (int j = k + 1; j < mat_size + 1; ++j){
                mpz_submul(coeff[i][j], scale, coeff[k][j]);
                mpz_mod(coeff[i][j], coeff[i][j], p);
            }
        }
    }
    GFLinearSolver(coeff, p);
}
void CoeffCal(mpz_t coeff[mat_size][mat_size + 1], const unsigned char *coeff_sr, const mpz_t *var_in, int len_in, int len_out, int coeff_amount, mpz_t modulus, int init_pos, int opt, int base, int hdeg, int len_mnml_base){
    mpz_set_str(mnml_base[0], "1", base);
    for (int i = 0; i < len_in; ++i)
        mpz_set(mnml_base[i + 1], var_in[i]);
    int left = 1;
    int right = len_in;
    int idx = right + 1;
    int iterator[len_in];
    for (int i = 0; i < len_in; ++i)
        iterator[i] = 1;
    for (int deg = 2; deg < hdeg; ++deg){
        for (int i = 0; i < len_in - opt; ++i){
            for (int j = left; j <= right; ++j){
                mpz_mul(mnml_base[idx], var_in[i], mnml_base[j]);
                mpz_mod(mnml_base[idx], mnml_base[idx], modulus);
                idx++;
            }
            left += iterator[i];
            unsigned int res = 0;
            for(int j = i; j < len_in; j++)
                res += iterator[j];
            iterator[i] = res;
        }
        iterator[len_in - 1] = 1 - opt;
        left = right + 1;
        right = idx - 1;
    }
    for (int eq_num = 0; eq_num < len_out; ++eq_num){
        int count = 0;
        int start_pos = init_pos + eq_num * coeff_amount;
        int left_tmp = left;
        int right_tmp = right;
        for (int i = 0; i < len_mnml_base; ++i){
            mpz_import(scale, 32, 1, sizeof(coeff_sr[0]), 0, 0, coeff_sr + (start_pos + count) * 32);
            mpz_addmul(coeff[eq_num][mat_size], mnml_base[i] ,scale);
            mpz_mod(coeff[eq_num][mat_size], coeff[eq_num][mat_size], modulus);
            count ++;
        }
        for (int i = 0; i < len_in; ++i){
            for (int j = left_tmp; j <= right_tmp; ++j){
                mpz_import(scale, 32, 1, sizeof(coeff_sr[0]), 0, 0, coeff_sr + (start_pos + count) * 32);
                mpz_mul(scale, scale, var_in[i]);
                mpz_mod(scale, scale, modulus);
                mpz_addmul(coeff[eq_num][mat_size], mnml_base[j], scale);
                mpz_mod(coeff[eq_num][mat_size], coeff[eq_num][mat_size], modulus);
                count ++;
            }
            left_tmp += iterator[i];
        }
        mpz_neg(coeff[eq_num][mat_size], coeff[eq_num][mat_size]);
        for (int l = 0; l < len_out; ++l){
            for (int i = 0; i < len_mnml_base; ++i){
                mpz_import(scale, 32, 1, sizeof(coeff_sr[0]), 0, 0, coeff_sr + (start_pos + count) * 32);
                mpz_addmul(coeff[eq_num][l], mnml_base[i], scale);
                mpz_mod(coeff[eq_num][l], coeff[eq_num][l], modulus);
                count ++;
            }
        }
    }
}
bool ECDSA_256_verify(const mpz_t r, const mpz_t s, const mpz_t message){
    if (mpz_cmp_ui(r, 0) <= 0 || mpz_cmp(r, n) >= 0 || mpz_cmp_ui(s, 0) <= 0 || mpz_cmp(s, n) >= 0){
        return FALSE;
    }
    mpz_invert(inv_s, s, n);
    mpz_set_ui(V1->x, 0);
    mpz_set_ui(V1->y, 0);
    mpz_set_ui(V2->x, 0);
    mpz_set_ui(V2->y, 0);
    mpz_mul(scalar, message, inv_s);
    mpz_mod(scalar, scalar, n);
    point_scalar(V1, G, scalar);
    mpz_mul(scalar, r, inv_s);
    mpz_mod(scalar, scalar, n);
    point_scalar(V2, pk, scalar);
    point_add(PP, V1, V2);
    if (point_is_infinity(PP)){
        return FALSE;
    }
    else{
        mpz_mod(PP->x, PP->x, n);
        if (mpz_cmp(PP->x, r) == 0){
            return TRUE;
        }
        else{
            return FALSE;
        }
    }
}
void ECDSA_256_sign(unsigned char sig_input[64], const unsigned char hash_input[32]){
    const char *n_str = "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";
    const char *p_str = "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff";
    const char *G_x_str = "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296";
    const char *G_y_str = "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5";
    const char *pk_x_str = "24A5B8DE827036C915D1A96D04C512EC548E45B36B48E16ADA083708512E2F69";
    const char *pk_y_str = "0FBA517095CD982552F6EDB78F3D4EF2800A767BD9469D3FA9E53E4AE2CF8537";
    unsigned char hash[32]; for(int i = 0; i < 32; i++){ hash[i] = hash_input[i];}
    point_init_set_str(G, G_x_str, G_y_str, 16);
    point_init_set_str(pk, pk_x_str, pk_y_str, 16);
    mpz_t hash_val, temp, temp2;
    mpz_t hash_val_const;
    mpz_init(hash_val);
    mpz_init(hash_val_const);
    mpz_import(hash_val, 32, 1, 1, 1, 0, hash);
    mpz_set(hash_val_const, hash_val);
    mpz_inits(temp, temp2, Q_y_neg, lambda, denominator, inv, scale, inv_s, scalar, NULL);
    for(int i = 0; i < 127; i++)
        mpz_init(mnml_base[i]);
    point_init(tmp);
    point_init(PP);
    point_init(V1);
    point_init(V2);
    mpz_init_set_str(n, n_str, 16);
    mpz_init_set_str(p, p_str, 16);
    mpz_init(a);
    mpz_sub_ui(a, p, 3);
    mpz_set_str(temp, "92696402816850788365912289725746782660095847965523055739723692989268172984508", 10);
    mpz_add(hash_val, hash_val, temp);
    mpz_set_str(temp, "81488770670818307", 10);
    mpz_powm(hash_val, hash_val, temp, n);
    mpz_set_str(temp, "64245627643302608474153", 10);
    mpz_powm(hash_val, hash_val, temp, n);
    mpz_set_str(temp, "19583213060460591175297", 10);
    mpz_powm(hash_val, hash_val, temp, n);
    mpz_set_str(temp, "677649073492583", 10);
    mpz_powm(hash_val, hash_val, temp, n);
    mpz_set_str(temp, "10227120287491701398896035081611795564320781186095923486182704828864950525244", 10);
    mpz_mul(hash_val, hash_val, temp);
    mpz_mod(hash_val, hash_val, n);
    mpz_t coeff[mat_size][mat_size + 1];
    for(int i = 0; i < mat_size; i++)
        for (int j = 0; j < mat_size + 1; j++)
            mpz_init(coeff[i][j]);
    mpz_t var_in_P[7];
    mpz_t var_in_N[5];
    for(int i = 0; i < 7; i++) mpz_init(var_in_P[i]);
    for(int i = 0; i < 5; i++) mpz_init(var_in_N[i]);
    mpz_set(var_in_P[0], hash_val);
    while(1){
        for(int i = 0; i < mat_size; i++) for (int j = 0; j < mat_size + 1; j++) mpz_set_ui(coeff[i][j], 0);
        for(int i = 2; i < 7; i++) mpz_set_ui(var_in_P[i], 0);
        for(int i = 0; i < 5; i++) mpz_set_ui(var_in_N[i], 0);
        mpz_set_ui(var_in_P[1], mpz_tstbit(hash_val, 0));
        for (int round = 0; round < 256; round++){
            for (int j = 0; j < mat_size; ++j)
                for (int k = 0; k < mat_size + 1; ++k)
                    mpz_set_ui(coeff[j][k], 0);
            if(round == 0) {
                CoeffCal(coeff, coeffs_p, var_in_P, 2, 6, 37, p, 0, 1, 62, 3, 5);
            }
            else if(round < 255) {
                mpz_set_ui(var_in_P[6], mpz_tstbit(hash_val, round));
                CoeffCal(coeff, coeffs_p, var_in_P, 7, 6, 322, p, 222 + 1932 * (round - 1), 1, 62, 3, 35);
            }
            else if(round == 255){
                mpz_set_ui(var_in_P[6], mpz_tstbit(hash_val, round));
                CoeffCal(coeff, coeffs_p, var_in_P, 7, 5, 854, p, 490950, 1, 62, 4, 112);
            }
            GaussElim(coeff, p);
            for(int i = 0; i < 6; i++)
                mpz_set(var_in_P[i], coeff[i][6]);
        }
        int I[5] = {0};
        for(int J = 0; J < 32768; J++){
            int X = list_overflows[J];
            for(int i = 0; i < 5; i++){
                I[i] = X%8;
                X /= 8;
            }
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 7; j++)
                    mpz_set_ui(coeff[i][j], 0);
            for (int i = 0; i < 5; i++){
                mpz_mul_ui(temp, p, I[i]);
                mpz_add(var_in_N[i], var_in_P[i], temp);
            }
            CoeffCal(coeff, coeffs_n, var_in_N, 5, 3, 1218, n, 0, 0, 62, 6, 252);
            GaussElim(coeff, n);
            if (ECDSA_256_verify(coeff[0][6], coeff[1][6], hash_val_const) == TRUE){
                #ifdef PRINT_GUESSES
                printf("(%d,%d,%d,%d,%d),\n", I[0],I[1],I[2],I[3],I[4]);
                #endif
                unsigned char sig[64];
                mpz_export(sig, NULL, 1, 32, 1, 0, coeff[0][6]);
                mpz_export(sig + 32, NULL, 1, 32, 1, 0, coeff[1][6]);
                for(int i = 0; i < 64; i++){ sig_input[i] = sig[i];}
                return;
            }
        }
    }
}
