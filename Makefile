# Makefile for building bispev C wrapper

# Compiler settings
CC = gcc
FC = gfortran
CFLAGS = -fPIC -O3 -Wall -I./include
FFLAGS = -fPIC -O3 -fno-underscoring
LDFLAGS = -shared

# Source files
FORTRAN_SRCS = src/fortran/bispev.f src/fortran/fpbisp.f src/fortran/fpbspl.f
C_SRCS = src/c/bispev_wrapper.c

# Object files
FORTRAN_OBJS = $(FORTRAN_SRCS:.f=.o)
C_OBJS = $(C_SRCS:.c=.o)
ALL_OBJS = $(FORTRAN_OBJS) $(C_OBJS)

# Target library
TARGET = libbispev.so

# Build rules
all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(FC) $(LDFLAGS) -o $@ $^ -lgfortran -lm

# Fortran compilation
%.o: %.f
	$(FC) $(FFLAGS) -c -o $@ $<

# C compilation
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean
clean:
	rm -f $(ALL_OBJS) $(TARGET)

# Install (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/lib/
	cp include/bispev_wrapper.h /usr/local/include/

.PHONY: all clean install