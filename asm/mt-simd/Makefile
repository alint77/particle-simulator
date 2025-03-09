# Makefile for N-body simulation with assembly optimization
CC = gcc
NASM = nasm
CFLAGS = -O3 -march=native -mavx2 -mfma -pthread -Wall -Wextra
NASMFLAGS = -f elf64

# Target executable
TARGET = particle_sim

# Source files
C_SOURCES = helper.c
ASM_SOURCES = compute.asm

# Object files
OBJECTS = $(C_SOURCES:.c=.o) $(ASM_SOURCES:.asm=.o)

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Assemble ASM source files
%.o: %.asm
	$(NASM) $(NASMFLAGS) $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)

# Run the simulation
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
