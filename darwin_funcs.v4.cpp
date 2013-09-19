/*
This is an anyD Darwin eom function with (fixed step) rk4 solver, for N particles.

Here, the C++ "valarray" is used to store the particle data. This container
has a few advantages. First, like an stl "vector", it deals with the dynamic memory allocation automatically, so
we don't have to worry about memory leaks. Second, valarrays are defined with several operator overloads that
are particularly useful (e.g. vector-scalar multiplication, and vector addition).

I have implemented valarrays of dimension one. They actually signify 2D arrays:
(Number of particles) /times (N coordinates + N momenta). I'm using what's called, "row-major order". With
this, a 2D (dimensions: rows*col) array is represented as a 1D array using the following transformation:

array2d[i][j] --> array1d[i*col + j].

In our case, col = (two /times number of spatial dimensions), and the number of rows is the (total number of particles).

While this is confusing at first, it does provide a considerable advantage. For one, these arrays are easier
to work with. Additionally, array elements that are continguous in memory can be accessed faster this way,
because of lack of caching.

Finally, I've defined these functions as "inline". This MIGHT make them faster ... but it's hard to say, really.

Anyway, I recommend that we try this out (see if it works), and then give a little more thought to
optimization.
*/

#include <iostream>
#include <fstream>
//valarrays require:
#include <valarray>

using namespace std;

/*
rk4 function prototype, where "r" is a valarray containing the x, y, ... and px, py, ... for each particle.
Use row-major ordering. "h" is the time step size, "t" is the current time, and "dim" is the number of
spatial dimensions. The final argument is the function which contains the eom. Note the number of arguments and their types.
*/
void inline rk4(valarray<double> &r, double h, double &t, const int dim, void (*f)(valarray<double> &, valarray<double> &, const int));
/*
Darwin EOM prototype. "k" is a valarray which contains the derivatives used in each step of rk4.
*/
void inline func(valarray<double> &r, valarray<double> &k, const int dim);
/*
1D Harmonic oscillator EOM prototype.
*/
void inline f_harmonic(valarray<double> &r, valarray<double> &k, const int dim);


int main()

{
    //I.C.s, loops, etc. Here's a sample using the 1D harmonic oscillator function, f_harmonic.

    /*
        //create output file
        ofstream OUTPUT("output.txt");

            //# of particles
        	const int N_p = 1;

            //# of space dims
        	const int dim = 1;

            //total valarray size
        	const int N_tot = 4*N_p*dim;

            //declare valarray r, to contain the p's and x's
        	valarray<double> r(N_tot);

            //I.C.s
        	r[0*dim+0] = 0.25;
        	r[(0+1)*dim+0] = 0.5;

        	//time steps, total time, etc.
    	    double sim_time = 100.0;
        	double dt = 0.001;
        	const int N_t = int(sim_time/dt);

            //start time
        	double t = 0.0;

            //loop until endtime, outputing data at each time step

        	for (int i = 0; i < N_t; i++)

        	{
        		for (int d = 0; d < dim; d++)

        	{
        		for (int i = 0; i < N_p; i++)

        		{
        		  OUTPUT << r[i*2*dim+d] << " " << r[(2*i+1)*dim+d] << endl;
        		}
        	}

        		rk4(r, dt, t, dim, f_harmonic);


        	}

        	//close file
        	OUTPUT.close();
    */

    return 0;

}



void inline rk4(valarray<double> &r, double h, double &t, const int dim, void (*f)(valarray<double> &, valarray<double> &, const int))

{
    //define k, r_step, and temp arrays to hold eom evaluations
    valarray<double> k(r.size());
    valarray<double> r_step(r.size());
    valarray<double> temp(r.size());

    const double half_h = h / 2.;

    //1st rk4 step
    f(r, k, dim);
    r_step = h * (1. / 6.) * k;
    temp = r + half_h * k;

    //2nd
    f(temp, k, dim);
    r_step += h * (1. / 3.) * k;
    temp = r + half_h * k;

    //3rd
    f(temp, k, dim);
    r_step += h * (1. / 3.) * k;
    temp = r + h * k;

    //4th
    f(temp, k, dim);

    //advance r in time
    r += r_step + h * (1. / 6.) * k;


    //advance time
    t += h;
}


void inline func(valarray<double> &r, valarray<double> &k, const int dim)

{
    //constants
    const double m = 1.0;
    const double c = 1.0;
    const double e = -1.0;
    const double fac = 1. / (2.* m * m * m * c * c);
    const double fac1 = 1. / (2.* m * m * c * c);

    int k_size = k.size();

    //reset k vector
    for (int i = 0; i < k_size; i++)
    {
        k[i] = 0.0;
    }

    //initialize vector to contain n_ij values
    double *n_ij = new double[dim];

    //get number of particles
    const int N_p = r.size() / (4 * dim);

    //arrays arrive in 1D form, where an element is identified by: i_par * dim + j_dim

    for (int i = 0; i < N_p; i++)

    {

        //calculate p_i^2
        double p_sq = 0.0;

        for (int d = 0; d < dim; d++)

        {
            p_sq += r[(2 * i + 1) * dim + d] * r[(2 * i + 1) * dim + d];
        }

        //evaluate equations of motion: -dH/dx = dp/dt; dH/dp = dx/dt
        for (int d = 0; d < dim; d++)
        {
            //dH/dp, 1st part
            k[i * 2 * dim + d] = r[(2 * i + 1) * dim + d] / m - p_sq * fac * r[(2 * i + 1) * dim + d];
        }

        //sum part of eom evaluation. Note: this probably needs a more efficient implementation
        for (int j = 0; j < N_p; j++)

        {
            double R_ij = 0.0;

            double pjdotn = 0.0;
            double pidotn = 0.0;
            double pidotpj = 0.0;


            //don't count ith particle
            if (i != j)

            {
                //calculate R_ij
                for (int d = 0; d < dim; d++)

                {
                    R_ij += (r[i * 2 * dim + d] - r[j * 2 * dim + d]) * (r[i * 2 * dim + d] - r[j * 2 * dim + d]);
                }

                R_ij = sqrt(R_ij);

                //calculate n_ij[d]
                for (int d = 0; d < dim; d++)

                {
                    n_ij[d] = (r[i * 2 * dim + d] - r[j * 2 * dim + d]) / R_ij;
                }

                //calculate some dot products
                for (int d = 0; d < dim; d++)

                {
                    pjdotn += n_ij[d] * r[(2 * j + 1) * dim + d];
                    pidotn += n_ij[d] * r[(2 * i + 1) * dim + d];
                    pidotpj += r[(2 * i + 1) * dim + d] * r[(2 * j + 1) * dim + d];
                }

                for (int d = 0; d < dim; d++)

                {
                    //rest of dH/dp
                    k[i * 2 * dim + d] -= (e * e / R_ij) * fac1 * (r[(2 * j + 1) * dim + d] + pjdotn * n_ij[d]);

                    //-dH/dx
                    double A = (e * e) / (R_ij * R_ij);
                    k[(2 * i + 1)*dim + d] += A * n_ij[d] * (1. - fac1 * pidotpj) - 3.*fac1 * A * n_ij[d] * pidotn * pjdotn;
                    k[(2 * i + 1)*dim + d] += fac1 * A * (r[(2 * i + 1) * dim + d] * pjdotn + r[(2 * j + 1) * dim + d] * pidotn);
                }

            }


        }

    }

    //free n_ij
    delete [] n_ij;
    n_ij = NULL;

}


void inline f_harmonic(valarray<double> &r, valarray<double> &k, const int dim)

{
    //constants
    const double m = 1.0;
    const double omega = 1.0;

    //get number of particles
    const int N_p = r.size() / (4 * dim);


    //reset k vector
    for (unsigned int i = 0; i < k.size(); i++)
    {
        k[i] = 0.0;
    }



    for (int d = 0; d < dim; d++)
    {

        for (int i = 0; i < N_p; i++)

        {
            k[(2 * i + 1)*dim + d] = (-1.) * m * omega * omega * r[i * 2 * dim + d];
            k[i * 2 * dim + d] = r[(2 * i + 1) * dim + d] / m;
        }

    }


}
