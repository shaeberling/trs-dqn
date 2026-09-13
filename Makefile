
.PHONY: all clean

# native/trs.c includes the Z80 core directly. Its generated opcode sources
# are already tracked in the submodule; a separate libz80.so is unnecessary.
all:
	$(MAKE) -C native

clean:
	$(RM) libtrs.so
