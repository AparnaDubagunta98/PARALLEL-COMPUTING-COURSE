// Game of Drones
//
// Find the drone in a grid 
//
// *** Lines marked "Do not remove" should be retained in your code ***
//
// Warning: Return values of calls are not checked for error to keep 
// the code simple.
//
// Requires drone.h to be in the same directory
// 
// Compilation command on ADA:
//
//   module load intel/2017A
//   icc -o drone.exe drone.c -lpthread -lrt
//
// Sample execution and output ($ sign is the shell prompt):
// 
// $ ./drone.exe 
//   Need four integers as input 
//   Use: <executable_name> <grid_size> <random_seed> <delay_nanosecs> <move_count>
// $ ./drone.exe 256 0 1000000 0
//   Drone = (111,195), success = 1, time (sec) =   3.4672
//
//
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include<math.h>

#include "drone.h"		// Do not remove
struct timespec start, stop; 	// Do not remove
//clock_t start, stop;
int main_seed ;

double total_time;		// Do not remove
int gridsize;			// Do not remove

unsigned int drone_x, drone_y; 	//Coordinates of drone (to be found)
int range_x, range_y;
// -------------------------------------------------------------------------
// Data structures and multithreaded code to find drone in the grid 
// ...
// ...
// ...
// ...
// ...
// ...

void find_drone(i){
    //with every move made, update location 
    //after all threads done, return final location

    int my_drone_x, my_drone_y;
    int my_seed = i + main_seed;
    srand48(my_seed); 
    my_drone_x = lrand48() % range_x;
    my_drone_y = lrand48() % range_y;


    printf("me as athread: %d,%d \n",my_drone_x,my_drone_y);

    int my_steps = check_grid(my_drone_x,my_drone_y);
    printf("my steps: %d \n",my_steps);
    printf("##########################\n");
}




// -------------------------------------------------------------------------
// Main program to find drone in a grid
int main(int argc, char *argv[]) {

    if (argc != 5) {
	printf("Need four integers as input \n"); 
	printf("Use: <executable_name> <grid_size> <random_seed> <delay_nanosecs> <move_count>\n"); 
	exit(0);
    }
    // Initialize grid
    gridsize = abs((int) atoi(argv[argc-4])); 		// Do not remove
    int seed = (int) atoi(argv[argc-3]); 		// Do not remove
    int delay_nsecs = abs((int) atoi(argv[argc-2]));// Do not remove
    int move_count = abs((int) atoi(argv[argc-1]));	// Do not remove
    initialize_grid(gridsize, seed, delay_nsecs, move_count); // Do not remove
    gridsize = get_gridsize();	 			// Do not remove
    main_seed = seed;

//    print_drone_path(); 
    range_x = range_y = ceil(sqrt(gridsize)); 
    printf("main \n");
    for(int i=0;i<gridsize;i++){
        find_drone(seed+i);
    }
    


//     clock_gettime(CLOCK_REALTIME, &start); 	// Do not remove
//     //start = clock();

//     // Multithreaded code to find drone in the grid 
//     // ...
//     // ...a
//     // ... sample serial code shown below ...
//     // ... works when drone is not allowed to move in check_grid() ...
    
    unsigned int i, j;
    int chk;
    for (i = 0; i < gridsize; i++) {
	    for (j = 0; j < gridsize; j++) {
	    chk = check_grid(i,j);
	    if (chk == 0) {
		drone_x = i; 
		drone_y = j; 
	    }
	}
    }
//     // ...
//     // ...
//     // ...

//     // Compute time taken
//     clock_gettime(CLOCK_REALTIME, &stop);			// Do not remove
//     total_time = (stop.tv_sec-start.tv_sec)			// Do not remove
// 	+0.000000001*(stop.tv_nsec-start.tv_nsec);		// Do not remove
// //stop = clock();
// //total_time = (1.0*stop-start)/CLOCKS_PER_SEC;

//     // Check if drone found, print time taken		
    printf("Drone = (%u,%u), success = %d ", // Do not remove
    drone_x, drone_y, check_drone_location(drone_x,drone_y));// Do not remove

    // Other code to wrap up things
    // ...
    // ...
    // ...

}

