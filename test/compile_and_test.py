# Source https://github.com/CryptoExperts/whibox_contest_submission_server
#!/usr/bin/env python3

import binascii
import json
import logging
import mmap
import os
import re
import subprocess
import sys
import traceback
# import urllib.request
# from urllib.parse import urljoin
from statistics import mean

# logging
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
logger = logging.getLogger()

CODE_SUCCESS = 0
ERR_CODE_CONTAININT_FORBIDDEN_STRING = 1
ERR_CODE_COMPILATION_FAILED = 2
ERR_CODE_BIN_TOO_LARGE = 3
ERR_CODE_LINK_FAILED = 4
ERR_CODE_EXECUTION_FAILED = 5
ERR_CODE_EXECUTION_EXCEED_RAM_LIMIT = 6
ERR_CODE_EXECUTION_EXCEED_TIME_LIMIT = 7

forbidden_strings = [b'#include', b'extern', b'_FILE__', b'__DATE__',
                     b'__TIME', b'__STDC_', b'__asm__', b'syscall']
forbidden_pattern = [re.compile(p) for p in [b'\sasm\W', ]]

# from docker-stack-prod.yml
os.environ["CHALLENGE_MAX_SOURCE_SIZE_IN_MB"] = "50"         # In MB. Must be identical in the launcher service. Must be 50 in production.
os.environ["CHALLENGE_MAX_MEM_COMPILATION_IN_MB"] = "500"    # In MB. Must be identical in the launcher service. Must be 500 in production.
os.environ["CHALLENGE_MAX_TIME_COMPILATION_IN_SECS"] = "100" # In seconds. Must be identical in the launcher service. Must be 100 in production.
os.environ["CHALLENGE_MAX_BINARY_SIZE_IN_MB"] = "20"         # In MB. Must be identical in the launcher service. Must be 20 in production.
os.environ["CHALLENGE_MAX_MEM_EXECUTION_IN_MB"] = "20"       # In MB. Must be identical in the launcher service. Must be 20 in production.
os.environ["CHALLENGE_MAX_TIME_EXECUTION_IN_SECS"] = "3"     # In seconds. Must be identical in the launcher service. Must be 1 in production.
os.environ["CHALLENGE_NUMBER_OF_TEST_VECTORS"] = "100"
os.environ['CHALLENGE_NUMBER_OF_TEST_EDGE_CASES'] = "3"

# TODO: CHANGE FILE_BASENAME to our implementation
os.environ['FILE_BASENAME'] = "dECDSA_ref"  # name of C source and executable
os.environ['UPLOAD_FOLDER'] = "./"  # folder of .c
MAIN_O_FOLDER = "./"
EXECUTE_FOLDER = "./"
# OBJECT_FOLDER = os.environ['UPLOAD_FOLDER']  # folder of .o
# EXECUTABLE_FOLDER = os.environ['UPLOAD_FOLDER']  # folder of ".exe"

assert "/" not in os.environ['FILE_BASENAME']

EXTRA_GCC_ARGS = None  # TODO: candidates "-Os","-O3","-flto"  (-flto 0.5MB reduction)


CHALLENGE_TEST_EDGE_CASES = [
    0,
    2**256 - 1,
    0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551,  # n
]
assert int(os.environ['CHALLENGE_NUMBER_OF_TEST_EDGE_CASES']) == len(CHALLENGE_TEST_EDGE_CASES)


def exit_after_notifying_launcher(code, post_data=None):
    print("CODE ERROR:", code)
    if post_data is not None:
        print(post_data)
    
    # avoiding network communication
    if post_data:
        post_data = json.dumps(post_data).encode('utf8')
        logger.info(f"POST: {post_data}")
    if code == ERR_CODE_EXECUTION_FAILED:
        os._exit(1)
    else:
        os._exit(0)
    
    # url_to_ping_back = os.environ['URL_TO_PING_BACK']
    # url = urljoin(url_to_ping_back, './%d' % code)
    # logger.info(f"Contacting {url}")
    #
    # post data
    # try:
    #     if post_data:
    #         post_data = json.dumps(post_data).encode('utf8')
    #         logger.info(f"POST: {post_data}")
    #     req = urllib.request.Request(
    #         url,
    #         data=post_data,
    #         headers={'content-type': 'application/json'}
    #     )
    #     urllib.request.urlopen(req)
    #
    #     if code == ERR_CODE_EXECUTION_FAILED:
    #         os._exit(1)
    #     else:
    #         os._exit(0)
    # except Exception as e:
    #     print(e)
    #     logger.error(f"!!! Could not contact {url}")
    #     os._exit(1)


def try_fetch_messages():
    import secrets
    number_of_tests = int(os.environ['CHALLENGE_NUMBER_OF_TEST_VECTORS'])

    # message for which v4 fails
    hex_string = "cae32a06eb9a4410040171c2e89abd9837d563b315c2a832958d9b970bfd9350"
    aux = bytes.fromhex(hex_string)
    
    for case in CHALLENGE_TEST_EDGE_CASES:
        aux += case.to_bytes(32, byteorder="big")

    aux += secrets.token_bytes(32*(number_of_tests-1))

    assert len(aux) == 32 * (number_of_tests + int(os.environ['CHALLENGE_NUMBER_OF_TEST_EDGE_CASES']))
    return aux
    # try:
    #     url = os.environ['URL_FOR_FETCHING_MESSAGES']
    #     logger.info(f"Contacting {url}")
    #     return urllib.request.urlopen(url).read()
    # except:
    #     logger.error("Could not fetch messages")
    #     os._exit(1)


def preprocess(source):
    with open(source, 'rb', 0) as f, \
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as contents:

        include_str = b'#include <gmp.h>\n'
        if (idx := contents.find(include_str)) != -1:
            logger.info("Remove #include <gmp.h>")
            contents = contents[:idx] + contents[idx+17:]

        for string in forbidden_strings:
            if contents.find(string) != -1:
                error_message = f"Forbidden string '{string.decode()}' found."
                logger.warning(error_message)
                post_data = {"error_message": error_message}
                exit_after_notifying_launcher(
                    ERR_CODE_CONTAININT_FORBIDDEN_STRING, post_data)

        for pattern in forbidden_pattern:
            match = re.search(pattern, contents)
            if match is not None:
                matched = match.group(0).decode()
                pattern = pattern.pattern.decode()
                error_message = (f"The string '{matched}' in the source code "
                                 f"matches forbidden pattern '{pattern}'.")
                logger.warning(error_message)
                post_data = {"error_message": error_message}
                exit_after_notifying_launcher(
                    ERR_CODE_CONTAININT_FORBIDDEN_STRING, post_data)


def compile(basename, source, obj):
    try:
        # max ram in KB
        max_ram = int(os.environ['CHALLENGE_MAX_MEM_COMPILATION_IN_MB']) * (2**10) + 5000  # noqa
        max_cpu_time = int(os.environ['CHALLENGE_MAX_TIME_COMPILATION_IN_SECS']) + 10  # noqa
        cmd_ulimit_ram = f'ulimit -v {max_ram}'
        cmd_ulimit_cpu_time = f'ulimit -t {max_cpu_time}'
        cmd_compile = f'gcc {"" if EXTRA_GCC_ARGS is None else EXTRA_GCC_ARGS} -c {source} -o {obj}'

        cmd_all = f'{cmd_ulimit_ram}; {cmd_ulimit_cpu_time}; {cmd_compile}'
        logger.info(f"Compilation CMD: {cmd_all}")
        compile_prcess = subprocess.run(cmd_all, check=True, shell=True,
                                        stderr=subprocess.PIPE)

        if b'warning: implicit declaration of function' in compile_prcess.stderr:
            err_msg = re.sub(
                r'/uploads/[a-z0-9]{32}\.c\:(\d+\:)*', '',
                compile_prcess.stderr.decode("utf-8"))
            post_data = {"error_message": err_msg}
            traceback.print_exc()
            exit_after_notifying_launcher(
                ERR_CODE_COMPILATION_FAILED, post_data)
    except Exception as e:
        logger.error(f"The compilation of file {basename}.c failed.")
        logger.error(f"The compile command is:\t{cmd_compile}\t")
        traceback.print_exc()
        exit_after_notifying_launcher(ERR_CODE_COMPILATION_FAILED)
    logger.info(f"The compilation of {basename}.c succeeded.")


def link(basename, obj, executable):
    try:
        # cmd_list = ['gcc', '/main.o', obj, '-lgmp', '-o', executable]
        cmd_list = ['gcc', MAIN_O_FOLDER + '/main.o', obj, '-lgmp', '-o', executable]
        if EXTRA_GCC_ARGS is not None:
            cmd_list = [cmd_list[0]] + EXTRA_GCC_ARGS.split(" ") + cmd_list[1:]
        subprocess.run(cmd_list, check=True)
    except:
        logger.error(f"The link of the file with basename {basename} failed.")
        exit_after_notifying_launcher(ERR_CODE_LINK_FAILED)
    logger.info(f"The link of the file with basename {basename} succeeded.")


def performance_measure(executable,
                        messages,
                        number_of_tests,
                        ram_limit,
                        cpu_time_limit):
    current_test_index = 0
    signatures = b''
    all_cpu_time = list()
    all_max_ram = list()

    to_break = False
    while current_test_index < number_of_tests:
        current_message = messages[
            current_test_index*32: (current_test_index+1)*32]
        current_message_as_text = binascii.hexlify(current_message).decode()
        cmd = (f'{EXECUTE_FOLDER}execute.py {executable} {current_message_as_text} '
               f'{current_test_index}')
        try:
            ps = subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                                shell=True)
            signature_as_text, cpu_time, ram = json.loads(ps.stdout)
            # ram = ram / 1000
            current_signature = bytes.fromhex(signature_as_text)
            if len(current_signature) != 64:
                raise Exception("Signature is too short")
            
            #
            from PUBLICKEY import pa_str
            from commands import ecdsa_verify_str
            if not ecdsa_verify_str(pa_str, current_message_as_text, signature_as_text):
                print(f"\n\nERROR: failed ecdsa_verify_str of {current_test_index}-th signature {signature_as_text} "
                      f"of message {current_message_as_text}\n\n")
                to_break = True
            
            signatures += current_signature

            # check whether we reach the limitation
            if cpu_time > cpu_time_limit:
                logger.warning("Execution reaches CPU time limit: "
                               f"{cpu_time:.2f}s was used!")
                post_data = {"cpu_time": cpu_time}
                # exit_after_notifying_launcher(
                #     ERR_CODE_EXECUTION_EXCEED_TIME_LIMIT,
                #     post_data=post_data)
                print(f"\n\nERROR: reached CPU time limit {cpu_time:.2f}s of {current_test_index}-th signature {signature_as_text}"
                      f" of message {current_message_as_text}\n\n")
                to_break = True
            if ram > ram_limit:
                logger.warning("Execution reaches memory limit: "
                               f"{ram/1024:.2f}MB")
                post_data = {"ram": ram/1024.}
                # exit_after_notifying_launcher(
                #     ERR_CODE_EXECUTION_EXCEED_RAM_LIMIT,
                #     post_data=post_data)
                print(f"\n\nERROR: reached memory limit {ram/1024:.2f}MB of {current_test_index}-th signature {signature_as_text}"
                      f" of message {current_message_as_text}\n\n")
                to_break = True

            all_cpu_time.append(cpu_time)
            all_max_ram.append(ram)
            current_test_index += 1
            if to_break:
                break
        except Exception as e:
            logger.error(f"Execution failed: {cmd}")
            traceback.print_exc()
            logger.error("===========")
            logger.error(e)
            exit_after_notifying_launcher(ERR_CODE_EXECUTION_FAILED)

    logger.info("The execution succeeded and we retrieved the signatures.")
    all_cpu_time.sort()
    all_max_ram.sort()
    if len(all_cpu_time) < 5:
        print(f"WARNING: small number of data points ({len(all_cpu_time)})")
        average_cpu_time = mean(all_cpu_time)
    else:
        average_cpu_time = mean(all_cpu_time[5:-5])
    if len(all_max_ram) < 5:
        print(f"WARNING: small number of data points ({len(all_max_ram)})")
        average_max_ram = mean(all_max_ram)
    else:
        average_max_ram = mean(all_max_ram[5:-5])

    return (signatures, average_cpu_time, average_max_ram)


def main():
    logger.info("Start compilation and test")

    # Compile
    upload_folder = os.environ['UPLOAD_FOLDER']
    basename = os.environ['FILE_BASENAME']
    source_file = basename + '.c'
    object_file = basename + '.o'
    path_to_source = os.path.join(upload_folder, source_file)
    path_to_object = os.path.join('/tmp', object_file)

    # check forbidden string and pattern
    logger.info("***** Preprocess the code *****")
    preprocess(path_to_source)

    logger.info("***** Start to compile *****")
    compile(basename, path_to_source, path_to_object)

    # check the binary size
    max_bin_size = 2**20 * int(os.environ['CHALLENGE_MAX_BINARY_SIZE_IN_MB'])
    bin_size = os.path.getsize(path_to_object)
    if bin_size > max_bin_size:
        exit_after_notifying_launcher(ERR_CODE_BIN_TOO_LARGE)

    # link
    logger.info("***** Start to link *****")
    path_to_executable = '/tmp/main'
    link(basename, path_to_object, path_to_executable)

    # fetch the plaintexts to encrypt
    logger.info("***** Start to fetch messages to sign *****")
    messages = try_fetch_messages()

    # performance measure
    logger.info("***** Sign messages, and measure performances *****")
    number_of_tests = int(os.environ['CHALLENGE_NUMBER_OF_TEST_VECTORS']) + \
        int(os.environ['CHALLENGE_NUMBER_OF_TEST_EDGE_CASES'])
    logger.info(f"Number of tests: {number_of_tests}")
    cpu_time_limit = int(os.environ['CHALLENGE_MAX_TIME_EXECUTION_IN_SECS'])
    ram_limit = 2**10 * int(os.environ['CHALLENGE_MAX_MEM_EXECUTION_IN_MB'])
    signatures, average_cpu_time, average_max_ram = performance_measure(
        path_to_executable, messages, number_of_tests,
        ram_limit, cpu_time_limit
    )
    size_factor = os.path.getsize(path_to_object) / max_bin_size
    ram_factor = average_max_ram * 1.0 / ram_limit
    time_factor = average_cpu_time * 1.0 / cpu_time_limit

    # If we reach this line, everything went fine
    # post_data = {
    #     "signatures": binascii.hexlify(signatures).decode(),
    #     "size_factor": size_factor,
    #     "ram_factor": ram_factor,
    #     "time_factor": time_factor
    # }
    # exit_after_notifying_launcher(CODE_SUCCESS, post_data=post_data)
    post_data = {
        "size_factor": "{:.2f}".format(size_factor),
        "ram_factor": "{:.2f}".format(ram_factor),
        "time_factor": "{:.2f}".format(time_factor),
        "total_size (MB)": "{:.2f}".format(os.path.getsize(path_to_object) * 1.0 / 2**20),
        "total_ram (MB)": "{:.2f}".format(average_max_ram * 1.0 / 2**10),
        "total_time (s)": "{:.2f}".format(average_cpu_time),
    }
    import pprint
    pprint.pprint(post_data)


if __name__ == "__main__":
    main()
 
