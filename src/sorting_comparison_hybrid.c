#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

// Sequential QuickSort
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

// Sequential BubbleSort
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

// OpenMP QuickSort Task 
void hybridQuickSortTask(int *arr, int low, int high) {
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

        #pragma omp task shared(arr)
        hybridQuickSortTask(arr, low, pi - 1);

        #pragma omp task shared(arr)
        hybridQuickSortTask(arr, pi + 1, high);
    }
}

void hybridQuickSort(int *arr, int n) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            hybridQuickSortTask(arr, 0, n - 1);
            #pragma omp taskwait
        }
    }
}

// OpenMP BubbleSort
void hybridBubbleSort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;
        #pragma omp parallel for shared(arr) reduction(|:swapped)
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                #pragma omp critical
                swap(&arr[j], &arr[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped)
            break;
    }
}

// MPI + OpenMP Hybrid Sort
void mpiHybridSort(int *data, int n, int rank, int size, void (*sortFunc)(int *, int)) {
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = n % size;
    int sum = 0;

    for (int i = 0; i < size; i++) {
        counts[i] = n / size + (i < rem ? 1 : 0);
        displs[i] = sum;
        sum += counts[i];
    }

    int local_n = counts[rank];
    int *local = malloc(local_n * sizeof(int));

    MPI_Scatterv(data, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    sortFunc(local, local_n);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0 && gathered != NULL) {
        hybridQuickSort(gathered, n);
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
    free(counts);
    free(displs);
}

// Accuracy checker
int calculate_accuracy(int *ref, int *test, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (ref[i] == test[i])
            correct++;
    }
    return (int)(((double)correct / n) * 100.0);
}

// Main function
int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    if (rank == 0) {
        printf("=== Sequential vs Hybrid(MPI+OpenMP) Sorting ===\n");
        printf("Enter array size: ");
        fflush(stdout);
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *original = malloc(n * sizeof(int));
    int *seqQuick = malloc(n * sizeof(int));
    int *seqBubble = malloc(n * sizeof(int));
    int *hybridQuick = malloc(n * sizeof(int));
    int *hybridBubble = malloc(n * sizeof(int));

    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < n; i++)
            original[i] = rand() % 100000;

        memcpy(seqQuick, original, n * sizeof(int));
        memcpy(seqBubble, original, n * sizeof(int));
        memcpy(hybridQuick, original, n * sizeof(int));
        memcpy(hybridBubble, original, n * sizeof(int));
    }

    double seqQuickTime = 0.0, seqBubbleTime = 0.0, hybridQuickTime = 0.0, hybridBubbleTime = 0.0;

    if (rank == 0) {
        double start = MPI_Wtime();
        seqQuickSort(seqQuick, 0, n - 1);
        seqQuickTime = MPI_Wtime() - start;

        if (n <= 50000) {
            start = MPI_Wtime();
            seqBubbleSort(seqBubble, n);
            seqBubbleTime = MPI_Wtime() - start;

            MPI_Bcast(hybridBubble, n, MPI_INT, 0, MPI_COMM_WORLD);
            start = MPI_Wtime();
            mpiHybridSort(hybridBubble, n, rank, size, hybridBubbleSort);
            hybridBubbleTime = MPI_Wtime() - start;
        }
    }

    MPI_Bcast(hybridQuick, n, MPI_INT, 0, MPI_COMM_WORLD);
    double start = MPI_Wtime();
    mpiHybridSort(hybridQuick, n, rank, size, hybridQuickSort);
    hybridQuickTime = MPI_Wtime() - start;

    if (rank == 0) {
        printf("\n=== Results (Array size: %d) ===\n", n);
        printf("--------------------------------------------------\n");
        printf("Algorithm           | Sequential | Hybrid (MPI+OMP)\n");
        printf("--------------------------------------------------\n");
        printf("QuickSort           | %9.6fs | %9.6fs\n", seqQuickTime, hybridQuickTime);

        if (n <= 50000)
            printf("BubbleSort          | %9.6fs | %9.6fs\n", seqBubbleTime, hybridBubbleTime);
        else
            printf("BubbleSort          |   ---    |   --- \n");

        printf("--------------------------------------------------\n");
        printf("\nAccuracy of Hybrid QuickSort: %d%%\n", calculate_accuracy(seqQuick, hybridQuick, n));
        if (n <= 50000)
            printf("Accuracy of Hybrid BubbleSort: %d%%\n", calculate_accuracy(seqBubble, hybridBubble, n));

        printf("\nFirst 100 Sorted Elements (QuickSort - Sequential): \n");
        for (int i = 0; i < 100 && i < n; i++)
            printf("%d ", seqQuick[i]);
        printf("\n");
    }

    free(original);
    free(seqQuick);
    free(seqBubble);
    free(hybridQuick);
    free(hybridBubble);

    MPI_Finalize();
    return 0;
}
