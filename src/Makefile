CC = gcc

#Sequential

#CFLAGS = -Wall -O2
#TARGET = sequential_sort
#SOURCE = sequential_sort.c

#OpenMP

#CFLAGS = -Wall -O2 -fopenmp
#TARGET = openmp_sort
#SOURCE = openmp_sort.c

#Comparision

CFLAGS = -Wall -O3 -fopenmp
TARGET = sorting_comparison
SOURCE = sorting_comparison.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o ../output/$(TARGET) $(SOURCE)

clean:
	rm -f ../output/$(TARGET)

.PHONY: all clean







