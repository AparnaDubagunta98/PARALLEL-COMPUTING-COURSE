// Header file for drone.c 
//
// *** THIS HEADER FILE SHOULD NOT BE MODIFIED ***
//
// Contains data structure to track the drone
//
// Contains following routines 
//
//    place_drone(unsigned int x, unsigned int y)
//      - place drone at an initial location (x,y)
//
//    move_drone()
//      - move drone to one of the four neighboring cells chosen randomly or 
//        to remain in its current location
//
//    check_grid(unsigned int x, unsigned int y)
//	- check grid location (i,j) to determine if drone is present
//        may move the drone (check code ...) 
//
//    initialize_grid(unsigned int n, unsigned int seed, 
//                    unsigned int nanosle    ep_ntime)
//	- initialize gridsize to n, places drone randomly at an initial 
//        location in the grid followed by calls to move_drone(),
//        and initializes maximum delay for check_grid() calls
//
//    get__MAX_PATH_LENGTH()
//      - returns _MAX_PATH_LENGTH, used to check return value from 
//        checck_grid()
//
//    get_gridsize()
//	- returns n where the grid is of size n x n
//
//    get_drone_location(unsigned int *x, unsigned int *y)
//	- used to get location of drone
// 
#include <stdlib.h>
#include <time.h>

// -------------------------------------------------------------------------
// Data structures for drone path and grid

#define _MAX_PATH_LENGTH 1000000      	// Maximum no. of steps drone can take
struct drone_path {
    unsigned int x[_MAX_PATH_LENGTH];  	// x-coordinate of cell on drone path
    unsigned int y[_MAX_PATH_LENGTH];  	// y-coordinate of cell on drone path
    unsigned int t[_MAX_PATH_LENGTH];  	// step number when drone was at this 
    				       	// cell
    unsigned int current;	       	// current step number
    unsigned int move_counter;	       	// count number of hits to drone path 
    				       	// since last drone move
};
struct drone_path drone;	       	// Drone path

unsigned int _grid_size = 4096;	       	// grid size, initialized to 
				       	// default value
struct timespec delay;		       	// forced delay at each check_grid() 
					// call
int move_freq;			       	// number of hits to the drone path 
				       	// before the drone can move one step

// -------------------------------------------------------------------------
// Routines for the grid

// Place drone at an initial location
// - Note that x and y are unsigned integers, therefore non-negative
void place_drone(unsigned int x, unsigned int y) {
    if ((x <= _grid_size-1) && (y <= _grid_size-1)) { // x, y are non-negative
        drone.current = 0; 
	drone.x[drone.current] = x;
	drone.y[drone.current] = y;
	drone.t[drone.current] = 0;
	drone.move_counter = 0;
    }
    else {
        printf("Invalid position to place drone. Aborting.\n"); 
	exit(0); 
    }
}

// Move drone to one of the four neighboring cells chosen randomly or remain 
// in its current location
// - Assumes _grid_size > 1; set in initialize_grid()
//
void move_drone() {
    if (drone.move_counter < move_freq) {
        // Do not move if number of hits to drone path below move_freq
	drone.move_counter++; 
    }
    else {
        // Move drone randomly to one of four neighbor cells
	drone.move_counter = 0;
	/*******************/
	if(drone.current == 0)
	{
		//initial position
		printf("drone at: %d , %d \n", drone.x[drone.current],drone.y[drone.current]);
	}
	/******************/
	if (drone.current < _MAX_PATH_LENGTH-1) {
	    drone.current++; 
	    drone.x[drone.current] = drone.x[drone.current-1]; 
	    drone.y[drone.current] = drone.y[drone.current-1]; 
	    drone.t[drone.current] = drone.t[drone.current-1]+1; 
	    switch (lrand48() % 4) {
		case 0: // Move right (left if reached grid edge)
			//printf("moving right \n");
		    if (drone.x[drone.current-1] < _grid_size-1) 
			drone.x[drone.current]++;
		    else 
			drone.x[drone.current]--;

		    //printf("drone at: %d , %d \n", drone.x[drone.current],drone.y[drone.current]);
		    break;
		case 1: // Move left (right if reached grid edge)
			//printf("moving left \n");
		    if (drone.x[drone.current-1] > 0) 
			drone.x[drone.current]--;
		    else 
			drone.x[drone.current]++;

		    //printf("drone at: %d , %d \n", drone.x[drone.current],drone.y[drone.current]);
		    break;
		case 2: // Move up (down if reached grid edge)
		    //printf("moving up \n");
		    if (drone.y[drone.current-1] < _grid_size-1) 
			drone.y[drone.current]++;
		    else 
			drone.y[drone.current]--;

		    //printf("drone at: %d , %d \n", drone.x[drone.current],drone.y[drone.current]);
		    break;
		case 3: // Move down (up if reached grid edge)
		    //printf("moving down \n");
		    if (drone.y[drone.current-1] > 0) 
			drone.y[drone.current]--;
		    else 
			drone.y[drone.current]++;

		    //printf("drone at: %d , %d \n", drone.x[drone.current],drone.y[drone.current]);
		    break;
	    }
	} 
	else {
	    printf("Drone has reached END-OF-LIFE!\n");
	    exit(0); 
	}
    }
}

// Check grid location (x,y)
// - Returns 0 if drone is at (x,y)
// - Returns _MAX_PATH_LENGTH+1 if drone was never at (x,y)
// - Returns the number of steps the drone has taken since the last time 
//   it was at (x,y) AND may move the drone to a neighboring cell or 
//   remain at its current position AND introduces delay
//
int check_grid(unsigned int x, unsigned int y) {
    nanosleep(&delay, NULL);
    int i = drone.current;
    while ((i >= 0) && ((x != drone.x[i]) || (y != drone.y[i]))) {
	i--;
    }
    if (i < 0) 
        // (x,y) not in drone path
	return _MAX_PATH_LENGTH+1;
    else {
        // (x,y) in drone path

//        move_drone();

	// return steps from (x,y) to drone's current location
	return (drone.t[drone.current] - drone.t[i]);   
    }
}

// Initialize grid size, drone location, and delay for each check_grid query
void initialize_grid(unsigned int n, unsigned int seed, int delay_nsecs, int move_count) {
    int i;
    // Initialize grid size
    if ((n > 1) && (n <= _grid_size)) {
	_grid_size = n;
    } else {
	printf("initialize_grid: Grid size set to %u (maximum allowed)\n", 
		_grid_size); 
    }
    // Initialize location of drone to random grid location
    srand48(seed); 
    place_drone(lrand48() % _grid_size, lrand48() % _grid_size); 

    for (i = 0; i < 16; i++) 
	move_drone(); 
    // Initialize move counter 
    move_freq = move_count;
    // Intialize delay
    delay.tv_sec = delay_nsecs/1000000000;
    delay.tv_nsec = 0;
}

// Determine _MAX_PATH_LENGTH 
int get_MAX_PATH_LENGTH() {
    return _MAX_PATH_LENGTH;
}

// Determine grid size 
int get_gridsize() {
    return _grid_size;
}

// Check drone location
int check_drone_location(unsigned int x, unsigned int y) {
    if ((x == drone.x[drone.current]) && (y == drone.y[drone.current])) 
	return 1;  
    else 
	return 0;
}

// Print drone path, for debugging
void print_drone_path() {
    int j;
    for (j = drone.current; j >= 0; j--) 
	printf("Drone Step:%d Location:(%u,%u)\n", j, drone.x[j],drone.y[j]);
}

