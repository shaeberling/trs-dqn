
CC ?= cc
CFLAGS ?= -O3

all: ../libtrs.so

../libtrs.so: trs.c ../libz80/z80.c ../libz80/z80.h $(wildcard ../libz80/codegen/opcodes*.c ../libz80/codegen/opcodes*.h)
	$(CC) $(CFLAGS) -I../libz80 -shared -fPIC -o $@.tmp trs.c
	mv $@.tmp $@
