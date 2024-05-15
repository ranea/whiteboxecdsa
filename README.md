# White-box Implementations of ECDSA

This repository contains C code and Python code of the two white-box implementations (a fast and less protected implementation and a slow and more protected implementation) of deterministic ECDSA signature algorithm on the NIST P-256 curve from the [WhibOx CHES 2021 Challenge](https://whibox.io/contests/2021/). These two white-box implementations are described in the paper [ECDSA White-Box Implementations: Attacks and Designs from CHES 2021 Challenge](https://tches.iacr.org/index.php/TCHES/article/view/9830).

In particular, this repository contains:
- C source code of the two white-box implementations in `C code` folder. Note that this is unobfuscated C source code.
- Python code of the two white-box implementations in `Python code\implementation_*.py`. These are alternative but functionally-equivalent implementations to the C ones. The Python implementations are also unobfuscated, they are slower than the C ones, but easier to understand.
- Python scripts to regenerate the C source code in `Python code\script_*.py`.

## Running the Python implementations

To run the Python implementations:

1. Ensure SageMath>=9 and Python 3 is installed.
2. Locate the following code block in `implementation_*.py` and introduce your ECDSA secret-public key pair:

```python
# secret key d (as integer) and public key Q = d G as hexadecimal string pair
SECRET_KEY = None  # e.g., 0x9C29EDDAEF2C2B4452052B668B83BE6365004278068884FA1AC3F6D0622875C3
Q_x_str = None  # e.g., "78E0E9DACCC47DE94D674DF3B35624A2F08E600B26B3444077022AD575AF4DB7"
Q_y_str = None  # e.g., "3084B4B8657EEA12396FDE260432BA7BDB3E092D61A42F830150D6CC8D798F9F"
```

3. Locate the following code block in `implementation_*.py` and introduce a random seed:

```python
# OTHER CONSTANTS
seed = None  # e.g., 0
```

4. Run `implementation_fast.py` or `implementation_slow.py`.


## Generating the C implementations

To generate the C implementations:

1. Ensure SageMath>=9 and Python >= 3.8 is installed.
2. Locate the following code block in `implementation_*.py` and introduce your ECDSA secret-public key pair:

```python
# secret key d (as integer) and public key Q = d G as hexadecimal string pair
SECRET_KEY = None  # e.g., 0x9C29EDDAEF2C2B4452052B668B83BE6365004278068884FA1AC3F6D0622875C3
Q_x_str = None  # e.g., "78E0E9DACCC47DE94D674DF3B35624A2F08E600B26B3444077022AD575AF4DB7"
Q_y_str = None  # e.g., "3084B4B8657EEA12396FDE260432BA7BDB3E092D61A42F830150D6CC8D798F9F"
```

3. Locate the following code block in `implementation_*.py` and introduce a random seed:

```python
# OTHER CONSTANTS
seed = None  # e.g., 0
```

4. Locate the line `FAST_MODE = None  # True or False` in `script_generate_overflows.py` and set `FAST_MODE` to `True` or `False`. Run `script_generate_overflows.py`

5. Locate the line `FAST_MODE = None  # True or False` in `script_printCarray.py` and set `FAST_MODE` to the same value as before. Run `script_printCarray.py`

6. Move `list_overflows*.c` and `equation*_binary.c` to the corresponding `C code\* version` folder. 

7. Run `C code\* version\script_merge.sh` and check the final C code `final_binary.c` with  `gcc -c final_binary.c -o final_binary.c -lgmp -Wno-builtin-declaration-mismatch -Wno-int-to-pointer-cast -Wno-trigraphs`.

## Testing the C implementations

To run and test the C implementation:

1. Install the python library click. 
2. Locate the following code block in `test\PUBLICKEY.py` and introduce your ECDSA public key pair:

```python
Q_x = None  # e.g., "78E0E9DACCC47DE94D674DF3B35624A2F08E600B26B3444077022AD575AF4DB7"
Q_y = None  # e.g., "3084B4B8657EEA12396FDE260432BA7BDB3E092D61A42F830150D6CC8D798F9F"
```

3. Move the final C code `final_binary.c` to `test` folder.
4. In `test\compile_and_test.py`, modify `os.environ['FILE_BASENAME']` with the name of the final C code (without the `.c` extension) containing `void ECDSA_256_sign`.
5. In macOS, uncomment the line `# ram = ram / 1000` of `compile_and_test.py` (in macOS the RAM is expressed in bytes while in Linux is in kilobytes)
6. Finally, compile the code and run the tests:

```bash
gcc -w -c main.c -o main.o
python3 compile_and_test.py
```

The `test` folder is from [GitHub - CryptoExperts/whibox\_contest\_submission\_server: Source code of the Whitebox Contest Submission Server](https://github.com/CryptoExperts/whibox_contest_submission_server).
