using System;
using System.Linq;

namespace csharp
{
    class Program
    {
        static void Main(string[] args)
        {
            OriginalVersion();
            LINQVersion();
        }

        // https://stackoverflow.com/questions/37934823/linear-regression-gradient-descent-using-c-sharp
        internal static void OriginalVersion()
        {
            Random rnd = new Random();
            const int N = 4;

            //We randomize the inital values of alpha and beta
            double theta1 = rnd.Next(0, 100);
            double theta2 = rnd.Next(0, 100);

            //Values of x, i.e the independent variable
            double[] x = new double[N] { 1, 2, 3, 4 };
            //VAlues of y, i.e the dependent variable
            double[] y = new double[N] { 5, 7, 9, 12 };
            double sumOfSquares1;
            double sumOfSquares2;
            double temp1;
            double temp2;
            double sum;
            double learningRate = 0.01;
            int count = 0;

            do
            {
                //We reset the Generalized cost function, called sum of squares 
                //since I originally used SS to 
                //determine if the function was minimized
                sumOfSquares1 = 0;
                sumOfSquares2 = 0;
                //Adding 1 to counter for each iteration to keep track of how 
                //many iterations are completed thus far
                count += 1;

                //First we calculate the Generalized cost function, which is
                //to be minimized
                sum = 0;
                for (int i = 0; i < N; i++)
                {
                    sum += Math.Pow((theta1 + theta2 * x[i] - y[i]), 2);
                }
                //Since we have 4 values of x and y we have 1/(2*N) = 1 /8 = 0.125
                sumOfSquares1 = 0.125 * sum;

                //Then we calcualte the new alpha value, using the derivative of 
                //the cost function. 
                sum = 0;
                for (int i = 0; i < N; i++)
                {
                    sum += theta1 + theta2 * x[i] - y[i];
                }
                //Since we have 4 values of x and y we have 1/(N) = 1 /4 = 0.25
                temp1 = theta1 - learningRate * 0.25 * sum;

                //Same for the beta value, it has a different derivative
                sum = 0;
                for (int i = 0; i < N; i++)
                {
                    sum += (theta1 + theta2 * x[i] - y[i]) * x[i];
                }

                temp2 = theta2 - learningRate * 0.25 * sum;

                //WE change the values of alpha an beta at the same time, otherwise the 
                //function wont work                
                theta1 = temp1;
                theta2 = temp2;

                //We then calculate the cost function again, with new alpha and beta values 
                sum = 0;
                for (int i = 0; i < N; i++)
                {
                    sum += Math.Pow((theta1 + theta2 * x[i]) - y[i], 2);
                }
                sumOfSquares2 = 0.125 * sum;

                Console.WriteLine("Alpha: {0:N}", theta1);
                Console.WriteLine("Beta: {0:N}", theta2);
                Console.WriteLine("GCF Before: {0:N}", sumOfSquares1);
                Console.WriteLine("GCF After: {0:N}", sumOfSquares2);
                Console.WriteLine("Iterations: {0}", count);
                Console.WriteLine(" ");

            } while (sumOfSquares2 <= sumOfSquares1 && count < 5000);
            //we end the iteration cycle once the generalized cost function
            //cannot be reduced any further or after 5000 iterations  
            Console.ReadLine();
        }

        internal static void LINQVersion(){
            Random rnd = new Random();
            double theta1 = rnd.Next(0, 100);
            double theta2 = rnd.Next(0, 100);

            double[] x = new double[] { 1, 2, 3, 4 };
            double[] y = new double[] { 5, 7, 9, 12 };

            double learningRate = 0.01;
            int iterations = 5000;

            for (int i = 0; i < iterations; i++)
            {
                double[] h_x = x.Select(item => theta2*item + theta1).ToArray();
                
                double theta2Derivative = (1.0/x.Length) * h_x.Zip(y, (h_x, y) => h_x - y).Zip(x, (z, x) => z * x).Sum();
                double theta1Derivative = (1.0/x.Length) * h_x.Zip(y, (h_x, y) => h_x - y).Sum();
                
                theta2 = theta2 - learningRate * theta2Derivative;
                theta1 = theta1 - learningRate * theta1Derivative;
            }

            Console.WriteLine("Slope: {0}", theta2);
            Console.WriteLine("Intercept: {0}", theta1);
        }
    }
}
