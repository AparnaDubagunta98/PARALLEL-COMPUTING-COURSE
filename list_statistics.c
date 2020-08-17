//
// Computes the minimum of a list using multiple threads
//
// Warning: Return values of calls are not checked for error to keep 
// the code simple.
//
// Compilation command on ADA ($ sign is the shell prompt):
//  $ module load intel/2017A
//  $ icc -o list_minimum.exe list_minimum.c -lpthread -lc -lrt
//
// Sample execution and output ($ sign is the shell prompt):
//  $ ./list_minimum.exe 1000000 9
// Threads = 9, minimum = 148, time (sec) =   0.0013
//
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_THREADS     65536
#define MAX_LIST_SIZE   268425456


int num_threads;		// Number of threads to create - user input 

int thread_id[MAX_THREADS];	// User defined id for thread
pthread_t p_threads[MAX_THREADS];// Threads
pthread_attr_t attr;		// Thread attributes 

pthread_mutex_t lock_stats;	// Protects mean,standard_deviation, count
pthread_cond_t cond_barrier; //barrier to wait for mean computation for threads to start stdev computation
//int minimum;			// Minimum value in the list
double mean, standard_deviation;
int count;			// Count of threads that have updated values

int list[MAX_LIST_SIZE];	// List of values
int list_size;			// List size

// Thread routine to compute mean of sublist assigned to thread; 
// update global value of mean 
void *find_stats (void *s) {
    int j;
    int my_thread_id = *((int *)s);
    //printf("thread: %d \n" , my_thread_id);

    int block_size = list_size/num_threads;
    int my_start = my_thread_id*block_size;
    int my_end = (my_thread_id+1)*block_size-1;
    if (my_thread_id == num_threads-1) my_end = list_size-1;

    //find local mean and standard deviation for sublist
    double my_sum = 0.000;
    for (j = my_start; j <= my_end; j++) 
    {
        //printf("j : %d \n",list[j]);
        my_sum += list[j];
        //printf("in for \n ");
    }
   // printf("in thread %f",my_sum);
    //first thread
    pthread_mutex_lock(&lock_stats);
    if(count == 0)
    {
        mean = my_sum;
    }
    else{
            mean += my_sum;
    }

    count ++;
    //printf(" thread %d, count %d \n",my_thread_id,count);
    if(count == num_threads)
    {
        //printf(" thread %d in end  \n ",my_thread_id);
        mean = mean / (list_size * 1.000);
        count = 0;
        pthread_cond_broadcast(&cond_barrier);
        pthread_mutex_unlock(&lock_stats);
    }
    else{
        //printf(" thread %d waiting \n ",my_thread_id);
        pthread_cond_wait(&cond_barrier,&lock_stats);
        pthread_mutex_unlock(&lock_stats);
        //printf("thread: %d  done waiting\n" , my_thread_id);

    }


    /*********** all threads done with mean calculation and start calculating stdev paralelly*********/

    //printf("thread: %d  came to stdev\n" , my_thread_id);
    double my_stdev_sum = 0.000;
    for (j = my_start; j <= my_end; j++) 
    {
        my_stdev_sum += pow((list[j] - mean),2)*1.000;
    }

    pthread_mutex_lock(&lock_stats);
    if(count == 0)
    {
        standard_deviation = my_stdev_sum;
    }
    else{
        standard_deviation += my_stdev_sum;
    }
    
    count++;

    if(count == num_threads)
    {
        standard_deviation = sqrt(standard_deviation / (list_size * 1.000));
        //printf("inside %f \n ", standard_deviation);
        pthread_cond_broadcast(&cond_barrier);
        pthread_mutex_unlock(&lock_stats);
    }
    else{
        pthread_cond_wait(&cond_barrier,&lock_stats);
        pthread_mutex_unlock(&lock_stats);
    }


    pthread_exit(NULL);

}

// Main program - set up list of randon integers and use threads to find
// the minimum value; assign minimum value to global variable called minimum
int main(int argc, char *argv[]) {

    struct timespec start, stop;
    double total_time, time_res;
    int i, j; 
    //int true_minimum;
    double true_mean, true_standard_deviation;
    double mean_sum = 0.000;
    double stdev_sum = 0.000;

    if (argc != 3) {
	printf("Need two integers as input \n"); 
	printf("Use: <executable_name> <list_size> <num_threads>\n"); 
	exit(0);
    }
    if ((list_size = atoi(argv[argc-2])) > MAX_LIST_SIZE) {
	printf("Maximum list size allowed: %d.\n", MAX_LIST_SIZE);
	exit(0);
    }; 
    if ((num_threads = atoi(argv[argc-1])) > MAX_THREADS) {
	printf("Maximum number of threads allowed: %d.\n", MAX_THREADS);
	exit(0);
    }; 
    if (num_threads > list_size) {
	printf("Number of threads (%d) < list_size (%d) not allowed.\n", num_threads, list_size);
	exit(0);
    }; 

    // Initialize mutex and attribute structures
    pthread_mutex_init(&lock_stats, NULL); 
    pthread_attr_init(&attr);
    pthread_cond_init(&cond_barrier, NULL); 
    //pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);



    // Initialize list, compute mean to verify result
    srand48(0); 	// seed the random number generator
    list[0] = lrand48(); 
    //list[0] = 0;
    mean_sum = list[0];
    for (j = 1; j < list_size; j++) {
	list[j] = lrand48(); 
    //list[j] = j;

    //********** serial calculation of mean **********
    mean_sum  += list[j];
    }
    //printf(" msum %f list size %d \n",mean_sum,list_size);

    true_mean = mean_sum / list_size *1.000;
   
    //********** serial calculation of standard deviation **********
    // //Compute true standard deviation
    for (j = 0; j < list_size; j++) {
        stdev_sum += pow((list[j] - true_mean),2)*1.000;
    }

    true_standard_deviation = sqrt(stdev_sum / list_size *1.000);



    //Initialize count
    count = 0;

    // Create threads; each thread executes find_minimum
    clock_gettime(CLOCK_REALTIME, &start);
    for (i = 0; i < num_threads; i++) {
	thread_id[i] = i; 
	pthread_create(&p_threads[i], &attr, find_stats, (void *) &thread_id[i]); 
    }
    // Join threads
    for (i = 0; i < num_threads; i++) {
	pthread_join(p_threads[i], NULL);
    }

     // printf("true mean %lf \n", true_mean);
     // printf("calc mean %lf \n", mean);
     // printf("true stdev %lf \n", true_standard_deviation);
     // printf("calc stdev %lf \n", standard_deviation);

    //printf(" join then min is: %d",minimum);

    //Compute time taken
    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec-start.tv_sec)
	+0.000000001*(stop.tv_nsec-start.tv_nsec);

    // Check answer


    // if (fabs(true_mean - mean) > 10e-5){
    // printf("Houston, we have a problem, MEAN INCORRECT!\n"); 
    // }

    // if (fabs(true_standard_deviation -standard_deviation) > 10e-5) {
    // printf("Houston, we have a problem, STANDARD DEVIATION INCORRECT!\n"); 
    // }

    // Print time taken
    printf("Threads = %d, mean = %f, standard_deviation= %f time (sec) = %8.4f\n", num_threads, mean,standard_deviation, total_time);

    // Destroy mutex and attribute structures
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&lock_stats);
    pthread_cond_destroy(&cond_barrier);
}

