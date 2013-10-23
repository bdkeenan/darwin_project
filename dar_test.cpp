/*
This is an anyD Darwin EOM solver using a fixed step rk4, for N_p/2 electrons and N_p/2 protons.
Included is a dipole interaction term, which is purported to be a model of the earth's magnetic field.

The I.C.s for position are randomly selected (uniform distrib.) from a region of space that has a size defined by
number density of the particles (at a distance of 1 A.U. from the Sun) divided by the total sim. particles.

The initial velocities are selected from a drifting Maxwellian distribution
(using the average solar wind velocity and temperature).

All particles begin motion along the x-axis at a min. distance of L*R_e from "earth" (R_e is the radius of the earth).

**********************************************************************************************************
NOTE: I question the appropriateness of these I.C.s. However, this is the best I think we can do (for now).
**********************************************************************************************************

I have implemented vectors of dimension one for the x's + p's. They actually signify 2D arrays:
(Number of particles) /times (N coordinates + N momenta). I'm using what's called, "row-major order". With
this, a 2D (dimensions: rows*col) array is represented as a 1D array using the following transformation:

array2d[i][j] --> array1d[i*col + j].

In this case, col = (two /times number of spatial dimensions), and the number of rows is the (total number of particles).

While this is confusing at first, it does provide a considerable advantage. For one, these arrays are easier
to work with. Additionally, array elements that are continguous in memory can be accessed faster this way,
because of lack of caching.

***********************************************************************************************************
In this code version, OpemMP is used to parallelize the outermost particle loop in
the EOM. As implemented, it will likely require tweaking for proper "load balancing", etc.

In particular, we need to find the best "schedule" ("dynamic", "guided", "static", etc.) and the best
"chunk" size.

When compiling this code for parallelization, we will need to include an openMP flag (e.g. -fopenmp, in g++).

I HIGHLY recommend compiling with a high level optimization flag (e.g. -O3, in g++).
***********************************************************************************************************
*/
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
//for openmp
#include <omp.h>
//std::setprecision
#include <iomanip>
//random number generators
#include "MersenneTwister.h"


//Include dipole term?
bool dip_yes = true;

//mass of electron (grams)
const double m_e = 9.10938291e-28;

//speed of light in vacuum (cm/s)
const double c = 29979245800.0;

//elementary electric charge (statcoulombs)
const double e = 4.80320425e-10;

//mass of proton (grams)
const double m_p = 1836.15267*m_e;

//radius of earth in cm
const double R_earth = 6.371e8;

//earth radii start
const double L = 10.0;

//avg. solar wind speed in cm/s
const double v_avg = 4.e7;

//time steps, total time, etc. (time in seconds)
const double sim_time = 2.0*L*R_earth/v_avg;
const double dt = 0.001;
const int N_t = int(sim_time/dt);

//# of particles (keep it even)
const int N_p = 500;

//# of space dims
const int dim = 3;

const int t_frame = std::floor(N_t/(25*15));

//Boltzmann Constant in erg/K
const double k_B = 1.3806488e-16;

//average solar wind energy (taking the temperature to be 8x10^5 K)
const double E_avg = k_B*(8.e5);


//For sanity's sake, I've chosen the magnetic moment along the z-axis. This is the dipole moment of earth in Gaussian units
const double m_z = 7.79e25;
const double m_z2 = m_z*m_z;

//avg. number density of electrons and protons (cm^-3)
const double n_avg = 6.85;

//constants for Darwin EOM
const double fac = 1./(2.*c*c);

//dipole term constants
const double fc = m_z/(c*c);
const double fc1 = 3.*fc;
const double fc2 = 2.*m_z/(c*c)*fc;
const double fc3 = 3.*fc2;


using namespace std;


/*
rk4 function prototype, where "r" is a vector containing the x, y, ... and px, py, ...
for each particle. Use row-major ordering. The vector "k" contains the time
derivatives (dx/dt, dp_x/dt, etc.). The time step size is "h", and "t" is the current
time. The penultimate argument is the function which contains the eom.
Note the number of arguments and their types.
*/
inline void rk4(std::vector<double> &r, double h, double &t, void (*f)(std::vector<double> &, std::vector<double> &, std::vector<double> &, std::vector<double> &), std::vector<double> &m, std::vector<double> &q);

//Darwin EOM prototype. "k" is a vector which contains the derivatives used in each step of rk4.
inline void func(std::vector<double> &r, std::vector<double> &k, std::vector<double> &m, std::vector<double> &q);

//This function will include the dipole term. Note: the dipole moment is along the z-axis
inline void dip_int(std::vector<double> &r, std::vector<double> &k, std::vector<double> &m, std::vector<double> &q);

//Initial Conditions
inline void ic(MTRand &mtrand, MTRand &mtrand1, double rmin, double rmax, std::vector<double> &r, std::vector<double> &m, std::vector<double> &q);


int main()

{
	//create output file
	ofstream OUTPUT("output.txt");

    //start all the particles at a certain dist. and distribute them randomly in a box
    //to get the correct "density"
	double xmin = L*R_earth;
	double xmax = xmin + pow(N_p/n_avg, 1./3.);

	//declare vector r, to contain the p's and x's
	std::vector<double> r(2*N_p*dim);

    //these vectors will contain the masses and charges
	std::vector<double> m(N_p);
	std::vector<double> q(N_p);


    //declare random number generators for I.C.s
	MTRand mtrand;
	MTRand mtrand1;

	mtrand.seed(1987);
	mtrand1.seed(1941);
	/*
	if these are commented out, the generators will be
	seeded with an array from /dev/urandom if available,
	otherwise they'll use a hash of time() and clock() values
	*/

	//Calculate I.C.s
	ic(mtrand, mtrand1, xmin, xmax, r, m, q);

	//start time
	double t = 0.0;

	//record current UNIX time
    time_t second0;
    second0 = time (NULL);


            //solve EOM for all time
        	for (int i = 0; i < N_t+1; i++)

        	{

        		for (int j = 0; j < N_p; j++)

        		{

			       //output positions (in earth radii) if they fall within the preset time interval
				   if (i % t_frame == 0)

				   {
					 OUTPUT << std::setprecision(10) << t << " " << r[j*2*dim]/R_earth << " " << r[2*j*dim+1]/R_earth << " " << r[2*j*dim+2]/R_earth << endl;
				   }

        		}


				//do runge-kutta to get r's for the next time step
        	    if (i < N_t)

        	    {
        		  rk4(r, dt, t, func, m, q);
        	    }

        	}

	time_t second1;
	second1 = time(NULL);
	std::cout << "effective run time: " << second1 - second0 << " second(s)." << std::endl;

	//close file
	OUTPUT.close();


    return 0;

}


inline void rk4(std::vector<double> &r, double h, double &t, void (*f)(std::vector<double> &, std::vector<double> &, std::vector<double> &, std::vector<double> &), std::vector<double> &m, std::vector<double> &q)

{
    const unsigned int r_size = r.size();

    //define k, r_step, and temp arrays to hold eom evaluations
	std::vector<double> k(r_size);
	std::vector<double> r_step(r_size);
	std::vector<double> temp(r_size);


	double half_h = h / 2.;
	double third_h = h / 3.;
	double sixth_h = h / 6.;

	//1st rk4 step
    f(r, k, m, q);

	for (unsigned int i = 0; i < r_size; i++)

	{
	  r_step[i] = sixth_h * k[i];
	  temp[i] = r[i] + half_h * k[i];
	}

	//2nd
    f(temp, k, m, q);


	for (unsigned int i = 0; i < r_size; i++)

	{
		r_step[i] += third_h * k[i];
		temp[i] = r[i] + half_h * k[i];
	}


	//3rd
    f(temp, k, m, q);


	for (unsigned int i = 0; i < r_size; i++)

	{
		r_step[i] += third_h * k[i];
		temp[i] = r[i] + h * k[i];
	}

	//4th
    f(temp, k, m, q);

	for (unsigned int i = 0; i < r_size; i++)

	{
		//advance r in time
		r[i] += r_step[i] + sixth_h * k[i];
	}


	//advance time
    t += h;


}


inline void func(std::vector<double> &r, std::vector<double> &k, std::vector<double> &m, std::vector<double> &q)

{

    unsigned int k_size = k.size();

    //reset k std::vector
    for (unsigned int i = 0; i < k_size; i++)
    {
        k[i] = 0.0;
    }


    //arrays arrive in 1D form, where an element is identified by: i_par * 2 * dim + j_dim

    for (int i = 0; i < N_p; i++)

    {
        //calculate p_i^2
        double p_sq = 0.0;

        int x_offseti = i * 2 * dim;
        int p_offseti = x_offseti + dim;

        for (int d = 0; d < dim; d++)

        {
            p_sq += r[p_offseti + d]*r[p_offseti + d];
        }

        //evaluate equations of motion: -dH/dx = dp/dt; dH/dp = dx/dt
        for (int d = 0; d < dim; d++)
        {
            //dH/dp, 1st part
            k[x_offseti + d] = r[p_offseti + d]/m[i] - fac*p_sq/(m[i]*m[i]*m[i])*r[p_offseti + d];
        }
    }


	    //some varibles for the EOM
        //(they must only be declared here, not initialized, for OpenMP)

	//declare std::vector pointer to contain n_ij values
	double *n_ij;

	double Rc_i, Rc_j, R_crit, R_ij, R_sq, pjdotn, pidotn, pidotpj, pi, pj, A, B, C;

	int j, d, x_offi, x_offj, p_offi, p_offj;


/*
this pragma instructs the program to do the outer for loop in openMP threads.
By default, all variables are shared among each thread, unless stated otherwise with a
private clause.

Since j, p_sq, etc. belong to a single particle (the "i"th particle), those variables are made private.
*/
#pragma omp parallel private(j,d,Rc_i,Rc_j,R_crit,R_ij,R_sq,pjdotn,pidotn,pidotpj,x_offi,x_offj,p_offi,p_offj,pi,pj,A,B,C,n_ij)

	{

	    n_ij = new double[dim];

//This schedule clause specifies the way the workload is determined.
//This may require some experimentation.
#pragma omp for schedule(static)

    for (int i = 0; i < N_p; i++)

    {
		 Rc_i = q[i]*q[i]/(m[i]*c*c);

         x_offi = i*2*dim;
         p_offi = x_offi + dim;

	    //sum part of eom evaluation. Note: this probably needs a more efficient implementation
        for (j = 0; j < i; j++)

        {
              x_offj = j*2*dim;
              p_offj = x_offj + dim;

              R_ij = 0.0;

              pjdotn = 0.0;
              pidotn = 0.0;
              pidotpj = 0.0;

                //calculate R_ij
                for (d = 0; d < dim; d++)

                {
                    R_sq = r[x_offi + d]-r[x_offj + d];
                    R_ij += R_sq*R_sq;
                }

                R_sq = R_ij;
			    R_ij = std::sqrt(R_ij);

			    Rc_j = q[j]*q[j]/(m[j]*c*c);

			    //critical radius
			    R_crit = Rc_i + Rc_j;


                if (R_ij < R_crit)
                {
                    R_ij = R_crit;
                    R_sq = R_crit*R_crit;
                }

                //calculate n_ij[d]
                for (d = 0; d < dim; d++)

                {
                   n_ij[d] = (r[x_offi + d]-r[x_offj + d])/R_ij;
                }

                //calculate some dot products
                for (d = 0; d < dim; d++)

                {
                    pi = r[p_offi + d];
                    pj = r[p_offj + d];

                    pidotn += n_ij[d]*pi;
                    pjdotn += n_ij[d]*pj;
                    pidotpj += pi*pj;
                }


                A = fac*q[i]*q[j]/(m[i]*m[j]*R_ij);
			    B = q[i]*q[j]/R_sq*(1.0-fac/(m[i]*m[j])*pidotpj);
			    C = A/R_ij;



                for (d = 0; d < dim; d++)

                {
                    //the rest of dH/dp
                    k[x_offi + d] -= A*(r[p_offj + d] + pjdotn*n_ij[d]);

                    //-dH/dx
                    k[p_offi + d] += B*n_ij[d] - 3.*C*n_ij[d]*pidotn*pjdotn;
                    k[p_offi + d] += C*(r[p_offi + d]*pjdotn + r[p_offj + d]*pidotn);
                }

        }

        for (j = i+1; j < N_p; j++)

              {
                    x_offj = j*2*dim;
                    p_offj = x_offj + dim;

                    R_ij = 0.0;

                    pjdotn = 0.0;
                    pidotn = 0.0;
                    pidotpj = 0.0;

                      //calculate R_ij
                      for (d = 0; d < dim; d++)

                      {
                          R_sq = r[x_offi + d]-r[x_offj + d];
                          R_ij += R_sq*R_sq;
                      }

                      R_sq = R_ij;
				      R_ij = std::sqrt(R_ij);

				      Rc_j = q[j]*q[j]/(m[j]*c*c);

				      //critical radius
				      R_crit = Rc_i + Rc_j;


                      if (R_ij < R_crit)
                      {
                          R_ij = R_crit;
                          R_sq = R_crit*R_crit;
                      }

                      //calculate n_ij[d]
                      for (d = 0; d < dim; d++)

                      {
                         n_ij[d] = (r[x_offi + d]-r[x_offj + d])/R_ij;
                      }

                      //calculate some dot products
                      for (d = 0; d < dim; d++)

                      {
                          pi = r[p_offi + d];
                          pj = r[p_offj + d];

                          pidotn += n_ij[d]*pi;
                          pjdotn += n_ij[d]*pj;
                          pidotpj += pi*pj;
                      }

				  A = fac*q[i]*q[j]/(m[i]*m[j]*R_ij);
				  B = q[i]*q[j]/R_sq*(1.0-fac/(m[i]*m[j])*pidotpj);
				  C = A/R_ij;


				  for (d = 0; d < dim; d++)

				  {
					  //the rest of dH/dp
					  k[x_offi + d] -= A*(r[p_offj + d] + pjdotn*n_ij[d]);

					  //-dH/dx
					  k[p_offi + d] += B*n_ij[d] - 3.*C*n_ij[d]*pidotn*pjdotn;
					  k[p_offi + d] += C*(r[p_offi + d]*pjdotn + r[p_offj + d]*pidotn);
				  }

                }

    }

    //free n_ij
    delete [] n_ij;

}

    n_ij = NULL;


	//dipole term
	if (dip_yes == true)

	{
		dip_int(r, k, m, q);
	}

}


inline void dip_int(std::vector<double> &r, std::vector<double> &k, std::vector<double> &m, std::vector<double> &q)

{

	for (int i = 0; i < N_p; i++)

	{
		int offset_x = i * 6;
		int offset_p = offset_x + 3;

		double x = r[offset_x];
		double y = r[offset_x + 1];
		double z = r[offset_x + 2];

		double px = r[offset_p];
		double py = r[offset_p + 1];

		double r_mag = std::sqrt(x*x + y*y + z*z);
		double r_mag2 = r_mag*r_mag;
		double r_mag3 = r_mag*r_mag2;
		double r_mag5 = r_mag3*r_mag2;
		//double r_mag6 = r_mag3*r_mag3;
		//double r_mag8 = r_mag6*r_mag2;

		//dH^{dip}_{int}/dp term
		double mult_fac = (q[i]/m[i])*fc/(r_mag3);

		k[offset_x] += mult_fac*y;
		k[offset_x + 1] -= mult_fac*x;

		//1st term in -dH^{dip}_{int}/dx

		k[offset_p] += mult_fac*py;
		k[offset_p + 1] += mult_fac*px;

		//2nd term in -dH^{dip}_{int}/dx
		mult_fac = (q[i]/m[i])*x*y*fc1/r_mag5;

		k[offset_p] += mult_fac*px;
		k[offset_p + 1] -= mult_fac*py;
	}

}



inline void ic(MTRand &mtrand, MTRand &mtrand1, double rmin, double rmax, std::vector<double> &r, std::vector<double> &m, std::vector<double> &q)

{
	//We'll choose equal parts electrons and protons:

	for (int i = 0; i < N_p; i++)

	{
		if (i % 2 == 0)

		{
		  q[i] = (-1.0)*e;
		  m[i] = m_e;
		}

		else

		{
		  q[i] = e;
		  m[i] = m_p;
		}
	}


	std::vector<double> v(N_p*dim);
	double r_range = rmax-rmin;

	for (int i = 0; i < N_p; i++)

	{
		int offset_x = 2*dim*i;
		int offset_p = offset_x + dim;

		r[offset_x] = rmin + mtrand.rand(r_range);
		r[offset_x+1] = mtrand.rand(r_range);
		r[offset_x+2] = mtrand.rand(r_range);

	  do

	    {

		 v[i*dim] = mtrand1.randNorm(v_avg, std::sqrt(E_avg/m[i]));

		} while (v[i*dim] > 0.1*c);

		v[i*dim+1] = 0.0;
		v[i*dim+2] = 0.0;

		r[offset_p] = (-1.0)*m[i]*v[i*dim]*(1.0+v[i*dim]*v[i*dim]/(2.0*c*c));
		r[offset_p+1] = 0.0;
		r[offset_p+2] = 0.0;


		if (dip_yes == true)

		{
			double r_vec3 = std::sqrt(r[offset_x]*r[offset_x] + r[offset_x+1]*r[offset_x+1] + r[offset_x+2]*r[offset_x+2]);
			r_vec3 = r_vec3*(r[offset_x]*r[offset_x] + r[offset_x+1]*r[offset_x+1] + r[offset_x+2]*r[offset_x+2]);

			r[offset_p] -= q[i]/(c*c)*m_z*r[offset_x+1]/r_vec3;
			r[offset_p+1] = q[i]/(c*c)*m_z*r[offset_x]/r_vec3;
		}


	}

    double *n_ij = new double[dim];

	for (int i = 0; i < N_p; i++)

	{
         double Rc_i = q[i]*q[i]/(m[i]*c*c);

         int xoff_i = i*2*dim;
         int poff_i = xoff_i + dim;

	    for (int j = 0; j < N_p; j++)

	    {
	         if (i != j)

	         {
                    int xoff_j = j*2*dim;

				    double R_ij = 0.0;
                    double vjdotn = 0.0;

                      //calculate R_ij
                      for (int d = 0; d < dim; d++)

                      {
                        R_ij += (r[xoff_i + d]-r[xoff_j + d])*(r[xoff_i + d]-r[xoff_j + d]);
                      }

				      R_ij = std::sqrt(R_ij);

				      double Rc_j = q[j]*q[j]/(m[j]*c*c);

				      double R_crit = Rc_i + Rc_j;

                      if (R_ij < R_crit)
                      {
                        R_ij = R_crit;
                      }

                      //calculate n_ij[d]
                      for (int d = 0; d < dim; d++)

                      {
                         n_ij[d] = (r[xoff_i + d]-r[xoff_j + d])/R_ij;
                         vjdotn += n_ij[d]*v[j*dim + d];
                      }


                      for (int d = 0; d < dim; d++)

                      {
                        r[poff_i + d] += fac*q[i]*q[j]/R_ij*(v[j*dim + d] + n_ij[d]*vjdotn);
                      }

			 }


		}

	}

	  //free n_ij
    delete [] n_ij;
    n_ij = NULL;

}
