rm -f final_fast_binary.c
cat equations_fast_binary.c list_overflows_fast.c logic.c > final_fast_binary.c
(echo "# pragma GCC diagnostic ignored \"-Wtrigraphs\"" && cat equations_fast_binary.c list_overflows_fast.c logic.c) > final_fast_binary.c
