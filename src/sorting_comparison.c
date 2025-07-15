#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

// Sequential QuickSort
void seqQuickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j <= high - 1; j++)
        {
            if (arr[j] < pivot)
            {
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

// Parallel QuickSort(openMP)
void parQuickSort(int arr[], int low, int high)
{
    const int THRESHOLD = 500; // avoid overhead for small segments
    if (low < high)
    {
        if (high - low < THRESHOLD)
        {
            seqQuickSort(arr, low, high);
            return;
        }

        // printf("Thread %d is sorting range [%d, %d] (size: %d)\n", omp_get_thread_num(), low, high, high - low + 1);

        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j <= high - 1; j++)
        {
            if (arr[j] < pivot)
            {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[high]);
        int pi = i + 1;

#pragma omp task shared(arr)
        parQuickSort(arr, low, pi - 1);

#pragma omp task shared(arr)
        parQuickSort(arr, pi + 1, high);

#pragma omp taskwait
    }
}

// Parallel QuickSort(MPI)
void mpiQuickSort(int *data, int n, int rank, int size)
{
    int chunk = n / size;
    int *local = malloc(chunk * sizeof(int));

    MPI_Scatter(data, chunk, MPI_INT, local, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Local sort
    seqQuickSort(local, 0, chunk - 1);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gather(local, chunk, MPI_INT, gathered, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Final merge (simple way)
        seqQuickSort(gathered, 0, n - 1);
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
}

void hybridQuickSort(int* data, int n, int rank, int size) {
    int chunk = n / size;
    int* local = malloc(chunk * sizeof(int));

    MPI_Scatter(data, chunk, MPI_INT, local, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Parallel sort with OpenMP
    #pragma omp parallel
    {
        #pragma omp single
        parQuickSort(local, 0, chunk - 1);
    }

    int* gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gather(local, chunk, MPI_INT, gathered, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        seqQuickSort(gathered, 0, n - 1); // simple final merge
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
}


// Sequential BubbleSort
void seqBubbleSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        int swapped = 0;
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(&arr[j], &arr[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped)
            break;
    }
}

// Parallel BubbleSort (openMP)
void parBubbleSort(int arr[], int n)
{
    int phase, i, temp;
    for (phase = 0; phase < n; phase++)
    {
#pragma omp parallel for private(temp) shared(arr)
        for (i = phase % 2; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }
    }
}

// Parallel BubbleSort (MPI)
void mpiBubbleSort(int *data, int n, int rank, int size)
{
    int chunk = n / size;
    int *local = malloc(chunk * sizeof(int));

    MPI_Scatter(data, chunk, MPI_INT, local, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Local sort
    seqBubbleSort(local, chunk);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gather(local, chunk, MPI_INT, gathered, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Final sort again
        seqBubbleSort(gathered, n);
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
}



int main()
{
    // initialize MPI
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    printf("=== Sorting Algorithm Comparison ===\n");
    printf("Enter array size: ");
    if (scanf("%d", &n) != 1 || n <= 0)
    {
        printf("Invalid array size!\n");
        return 1;
    }

    int *original = malloc(n * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < n; i++)
        original[i] = rand() % 100000;

    printf("\n=== Original Array ( First 100 Eleements): %d) ===\n", n);
    for (int i = 0; i < 100 && i < n; i++)
    {
        printf("%d ", original[i]);
    }
    printf("\n");

    int *seqQuick = malloc(n * sizeof(int));
    int *parQuick = malloc(n * sizeof(int));
    int *mpiQuick = malloc(n * sizeof(int));
    int *hybridQuick = malloc(n * sizeof(int));
    int *seqBubble = malloc(n * sizeof(int));
    int *parBubble = malloc(n * sizeof(int));
    int *mpiBubble = malloc(n * sizeof(int));
    int *hybridBubble = malloc(n * sizeof(int));

 
    
    memcpy(seqQuick, original, n * sizeof(int));
    memcpy(parQuick, original, n * sizeof(int));
    memcpy(seqBubble, original, n * sizeof(int));
    memcpy(parBubble, original, n * sizeof(int));

    double start, end;
    double seqQ_time, parQ_time, seqB_time, parB_time;

    // Sequential QuickSort
    start = omp_get_wtime();
    seqQuickSort(seqQuick, 0, n - 1);
    end = omp_get_wtime();
    seqQ_time = end - start;

    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     {
    //         printf("Total threads: %d\n", omp_get_num_threads());
    //     }
    // }

    // Parallel QuickSort
    start = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        parQuickSort(parQuick, 0, n - 1);
    }
    end = omp_get_wtime();
    parQ_time = end - start;

    if (n <= 50000)
    {
        // Sequential BubbleSort
        start = omp_get_wtime();
        seqBubbleSort(seqBubble, n);
        end = omp_get_wtime();
        seqB_time = end - start;
    }

    // Parallel BubbleSort
    start = omp_get_wtime();
    parBubbleSort(parBubble, n);
    end = omp_get_wtime();
    parB_time = end - start;

    printf("\n=== Results (Array size: %d) ===\n", n);
    printf("----------------------------------------\n");
    printf("Algorithm          | Sequential | Parallel\n");
    printf("----------------------------------------\n");
    printf("QuickSort          | %9.6fs | %9.6fs\n", seqQ_time, parQ_time);
    if (n <= 50000)
    {
        printf("BubbleSort         | %9.6fs | %9.6fs\n", seqB_time, parB_time);
    }
    else
    {
        printf("BubbleSort         | ------- s | %9.6fs\n", parB_time);
    }

    printf("----------------------------------------\n\n");

    printf("Performance Comparison:\n");

    // QuickSort
    if (seqQ_time < parQ_time)
    {
        printf("QuickSort: Sequential is faster (%.6fs vs %.6fs)\n", seqQ_time, parQ_time);
    }
    else if (parQ_time < seqQ_time)
    {
        printf("QuickSort: Parallel is faster (%.6fs vs %.6fs)\n", parQ_time, seqQ_time);
    }
    else
    {
        printf("QuickSort: Both have same performance (%.6fs)\n", seqQ_time);
    }

    // BubbleSort
    if (n <= 50000)
    {
        if (seqB_time < parB_time)
        {
            printf("BubbleSort: Sequential is faster (%.6fs vs %.6fs)\n", seqB_time, parB_time);
        }
        else if (parB_time < seqB_time)
        {
            printf("BubbleSort: Parallel is faster (%.6fs vs %.6fs)\n", parB_time, seqB_time);
        }
        else
        {
            printf("BubbleSort: Both have same performance (%.6fs)\n", seqB_time);
        }
    }
    else
    {
        printf("BubbleSort: Parallel is faster (%.6fs )\n", parB_time);
    }

    // Display few sorted values to verify correctness
    printf("\nFirst 100 Sorted Elements (QuickSort - Sequential): \n");
    for (int i = 0; i < 100 && i < n; i++)
    {
        printf("%d ", seqQuick[i]);
    }
    printf("\n");

    free(original);
    free(seqQuick);
    free(parQuick);
    free(seqBubble);
    free(parBubble);

    return 0;
}
