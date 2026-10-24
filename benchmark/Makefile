# Builds the benchmark, which runs on Linux only
CC = cc
CFLAGS = -O2 -Wall -Wextra
LDLIBS = -lpthread

cpupowerbench: main.c usage.c load.c load.h usage.h
	$(CC) $(CFLAGS) -o $@ main.c usage.c load.c $(LDLIBS)

clean:
	rm -f cpupowerbench

.PHONY: clean
