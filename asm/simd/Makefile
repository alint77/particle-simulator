# Compiler and assembler settings
CC=gcc
NASM=nasm
CFLAGS=-Wall -O2 -g
LDFLAGS=-lm

# Object files
OBJECTS=helper.o compute.o

# Target executable
TARGET=particle_sim

all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile C source files
helper.o: helper.c particles.h
	$(CC) $(CFLAGS) -c helper.c

# Assemble NASM source files
compute.o: compute.asm
	$(NASM) -f elf64 -g compute_simd.asm -o compute.o -O2

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean
