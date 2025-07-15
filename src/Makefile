# === Compiler selection ===
GCC = gcc
MPICC = mpicc

# === Flags ===
CFLAGS_OMP = -Wall -O3 -fopenmp
CFLAGS_MPI = -Wall -O3
CFLAGS_HYBRID = -Wall -O3 -fopenmp

# === Targets and sources ===
OUTPUT_DIR = ../output
$(shell mkdir -p $(OUTPUT_DIR))

OMP_TARGET = $(OUTPUT_DIR)/sorting_comparison_openmp
OMP_SRC = sorting_comparison_openmp.c

MPI_TARGET = $(OUTPUT_DIR)/sorting_comparison_mpi
MPI_SRC = sorting_comparison_mpi.c

HYBRID_TARGET = $(OUTPUT_DIR)/sorting_comparison_hybrid
HYBRID_SRC = sorting_comparison_hybrid.c

# === Default target ===
all: $(OMP_TARGET) $(MPI_TARGET) $(HYBRID_TARGET)

# === Build Rules ===
$(OMP_TARGET): $(OMP_SRC)
	$(GCC) $(CFLAGS_OMP) -o $@ $<

$(MPI_TARGET): $(MPI_SRC)
	$(MPICC) $(CFLAGS_MPI) -o $@ $<

$(HYBRID_TARGET): $(HYBRID_SRC)
	$(MPICC) $(CFLAGS_HYBRID) -o $@ $<

# === Run Targets ===
run-omp: $(OMP_TARGET)
	$(OMP_TARGET)

run-mpi: $(MPI_TARGET)
	mpirun -np 4 $(MPI_TARGET)

run-hybrid: $(HYBRID_TARGET)
	export OMP_NUM_THREADS=2 && mpirun -np 2 $(HYBRID_TARGET)

# === Clean target ===
clean:
	rm -f $(OUTPUT_DIR)/sorting_comparison_*

.PHONY: all clean run-omp run-mpi run-hybrid
