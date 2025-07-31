# === Compiler selection ===
GCC = gcc
MPICC = mpicc

# === Flags ===
CFLAGS_OMP    = -Wall -O3 -fopenmp
CFLAGS_MPI    = -Wall -O3
CFLAGS_HYBRID = -Wall -O3 -fopenmp

# === Directories ===
OUTPUT_DIR = ../output
$(shell mkdir -p $(OUTPUT_DIR))

# === Target Executables ===
OMP_TARGET       = $(OUTPUT_DIR)/sorting_comparison_openmp
MPI_Q_TARGET     = $(OUTPUT_DIR)/sorting_comparison_mpi_q
MPI_B_TARGET     = $(OUTPUT_DIR)/sorting_comparison_mpi_b
HYBRID_Q_TARGET  = $(OUTPUT_DIR)/sorting_comparison_hybrid_q
HYBRID_B_TARGET  = $(OUTPUT_DIR)/sorting_comparison_hybrid_b

# === Source Files ===
OMP_SRC       = sorting_comparison_openmp.c
MPI_Q_SRC     = sorting_comparison_mpi_q.c
MPI_B_SRC     = sorting_comparison_mpi_b.c
HYBRID_Q_SRC  = sorting_comparison_hybrid_q.c
HYBRID_B_SRC  = sorting_comparison_hybrid_b.c

# === Default build target ===
all: $(OMP_TARGET) $(MPI_Q_TARGET) $(MPI_B_TARGET) $(HYBRID_Q_TARGET) $(HYBRID_B_TARGET)

# === Build Rules ===
$(OMP_TARGET): $(OMP_SRC)
	$(GCC) $(CFLAGS_OMP) -o $@ $<

$(MPI_Q_TARGET): $(MPI_Q_SRC)
	$(MPICC) $(CFLAGS_MPI) -o $@ $<

$(MPI_B_TARGET): $(MPI_B_SRC)
	$(MPICC) $(CFLAGS_MPI) -o $@ $<

$(HYBRID_Q_TARGET): $(HYBRID_Q_SRC)
	$(MPICC) $(CFLAGS_HYBRID) -o $@ $<

$(HYBRID_B_TARGET): $(HYBRID_B_SRC)
	$(MPICC) $(CFLAGS_HYBRID) -o $@ $<

# === Run Targets ===
run-omp: $(OMP_TARGET)
	OMP_NUM_THREADS=4 $(OMP_TARGET)

run-mpi-q: $(MPI_Q_TARGET)
	mpirun -np 4 $(MPI_Q_TARGET)

run-mpi-b: $(MPI_B_TARGET)
	mpirun -np 4 $(MPI_B_TARGET)

run-hybrid-q: $(HYBRID_Q_TARGET)
	export OMP_NUM_THREADS=2 && mpirun -np 2 $(HYBRID_Q_TARGET)

run-hybrid-b: $(HYBRID_B_TARGET)
	export OMP_NUM_THREADS=2 && mpirun -np 2 $(HYBRID_B_TARGET)

# === Clean Target ===
clean:
	rm -f $(OUTPUT_DIR)/sorting_comparison_*

.PHONY: all clean run-omp run-mpi-q run-mpi-b run-hybrid-q run-hybrid-b
