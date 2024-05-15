rm -f final_binary.c
cat equations_binary.c list_overflows.c logic.c > final_binary.c
(echo "# pragma GCC diagnostic ignored \"-Wtrigraphs\"" && cat equations_binary.c list_overflows.c logic.c) > final_binary.c
