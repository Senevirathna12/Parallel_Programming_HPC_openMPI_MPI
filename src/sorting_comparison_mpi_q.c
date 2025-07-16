#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

// ----- Common Utilities -----
void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

void seqQuickSort(int arr[], int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[high]);
        int pi = i + 1;
        seqQuickSort(arr, low, pi - 1);
        seqQuickSort(arr, pi + 1, high);
    }
}

void mpiQuickSort(int *data, int n, int rank, int size) {
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = n % size, sum = 0;

    for (int i = 0; i < size; i++) {
        counts[i] = n / size;
        if (rem > 0) {
            counts[i]++;
            rem--;
        }
        displs[i] = sum;
        sum += counts[i];
    }

    int local_n = counts[rank];
    int *local = malloc(local_n * sizeof(int));

    MPI_Scatterv(data, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    seqQuickSort(local, 0, local_n - 1);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        seqQuickSort(gathered, 0, n - 1);
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
    free(counts);
    free(displs);
}

double calculate_accuracy(int *ref, int *test, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (ref[i] == test[i])
            correct++;
    }
    return ((double)correct / n) * 100.0;
}

int main(int argc, char **argv) {
    int rank, size, n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("=== Sequential vs MPI QuickSort Comparison ===\n");
        printf("Enter array size: ");
        fflush(stdout);
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *original = malloc(n * sizeof(int));
    int *seqQuickSorted = malloc(n * sizeof(int));
    int *mpiQuickSorted = malloc(n * sizeof(int));

    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < n; i++)
            original[i] = rand() % 100000;

        memcpy(seqQuickSorted, original, n * sizeof(int));
        memcpy(mpiQuickSorted, original, n * sizeof(int));

        printf("\n=== Original Array (First 100 elements): ===\n");
        for (int i = 0; i < 100 && i < n; i++)
            printf("%d ", original[i]);
        printf("\n");
    }

    double seq_q_time = 0.0, mpi_q_time = 0.0;

    if (rank == 0) {
        double start = MPI_Wtime();
        seqQuickSort(seqQuickSorted, 0, n - 1);
        seq_q_time = MPI_Wtime() - start;
    }

    MPI_Bcast(mpiQuickSorted, n, MPI_INT, 0, MPI_COMM_WORLD);
    double start = MPI_Wtime();
    mpiQuickSort(mpiQuickSorted, n, rank, size);
    mpi_q_time = MPI_Wtime() - start;

    if (rank == 0) {
        printf("\n=== Results (Array size: %d) ===\n", n);
        printf("----------------------------------------\n");
        printf("Algorithm          | Sequential | MPI\n");
        printf("----------------------------------------\n");
        printf("QuickSort          | %9.6fs | %9.6fs\n", seq_q_time, mpi_q_time);
        printf("----------------------------------------\n");

        if (seq_q_time < mpi_q_time)
            printf("QuickSort: Sequential is faster (%.6fs vs %.6fs)\n", seq_q_time, mpi_q_time);
        else if (mpi_q_time < seq_q_time)
            printf("QuickSort: MPI is faster (%.6fs vs %.6fs)\n", mpi_q_time, seq_q_time);
        else
            printf("QuickSort: Both have same performance (%.6fs)\n", seq_q_time);

        double acc_quick = calculate_accuracy(seqQuickSorted, mpiQuickSorted, n);
        printf("\nAccuracy of MPI QuickSort (compared to Sequential): %.2f%%\n", acc_quick);

        printf("\nFirst 100 Sorted Elements (QuickSort - Sequential): \n");
        for (int i = 0; i < 100 && i < n; i++)
            printf("%d ", seqQuickSorted[i]);
        printf("\n");
    }

    free(original);
    free(seqQuickSorted);
    free(mpiQuickSorted);

    MPI_Finalize();
    return 0;
}
