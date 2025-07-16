#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

void seqBubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped)
            break;
    }
}

void mpiBubbleSort(int *data, int n, int rank, int size) {
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = n % size;
    int sum = 0;

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

    seqBubbleSort(local, local_n);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        seqBubbleSort(gathered, n);
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
        printf("=== Sequential vs MPI BubbleSort Comparison ===\n");
        printf("Enter array size: ");
        fflush(stdout);
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *original = malloc(n * sizeof(int));
    int *seqBubbleSorted = malloc(n * sizeof(int));
    int *mpiBubbleSorted = malloc(n * sizeof(int));

    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < n; i++)
            original[i] = rand() % 100000;

        memcpy(seqBubbleSorted, original, n * sizeof(int));
        memcpy(mpiBubbleSorted, original, n * sizeof(int));

        printf("\n=== Original Array (First 100 elements): ===\n");
        for (int i = 0; i < 100 && i < n; i++)
            printf("%d ", original[i]);
        printf("\n");
    }

    double seq_b_time = 0.0, mpi_b_time = 0.0;

    if (rank == 0) {
        double start = MPI_Wtime();
        seqBubbleSort(seqBubbleSorted, n);
        seq_b_time = MPI_Wtime() - start;
    }

    MPI_Bcast(mpiBubbleSorted, n, MPI_INT, 0, MPI_COMM_WORLD);
    double start = MPI_Wtime();
    mpiBubbleSort(mpiBubbleSorted, n, rank, size);
    mpi_b_time = MPI_Wtime() - start;

    if (rank == 0) {
        printf("\n=== Results (Array size: %d) ===\n", n);
        printf("----------------------------------------\n");
        printf("Algorithm          | Sequential | MPI\n");
        printf("----------------------------------------\n");
        printf("BubbleSort         | %9.6fs | %9.6fs\n", seq_b_time, mpi_b_time);
        printf("----------------------------------------\n");

        if (seq_b_time < mpi_b_time)
            printf("BubbleSort: Sequential is faster (%.6fs vs %.6fs)\n", seq_b_time, mpi_b_time);
        else if (mpi_b_time < seq_b_time)
            printf("BubbleSort: MPI is faster (%.6fs vs %.6fs)\n", mpi_b_time, seq_b_time);
        else
            printf("BubbleSort: Both have same performance (%.6fs)\n", seq_b_time);

        double acc_bubble = calculate_accuracy(seqBubbleSorted, mpiBubbleSorted, n);
        printf("\nAccuracy of MPI BubbleSort (compared to Sequential): %.2f%%\n", acc_bubble);

        printf("\nFirst 100 Sorted Elements (BubbleSort - Sequential): \n");
        for (int i = 0; i < 100 && i < n; i++)
            printf("%d ", seqBubbleSorted[i]);
        printf("\n");
    }

    free(original);
    free(seqBubbleSorted);
    free(mpiBubbleSorted);

    MPI_Finalize();
    return 0;
}
