using System;
using XPlot.Plotly;
using System.Linq;
using Deedle;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace LinearRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            // Prepare data
            var data = PrepareInputData("ex1data1.txt");
            Matrix<double> X = data.X;
            Vector<double> y = data.y;
            Console.WriteLine(X);
            Console.WriteLine(y);

            // ComputeCost
            var thetaSeries = new Series<int, double>(new int[2] { 0, 1 }, new double[2]);
            double J = ComputeCost(X, y, thetaSeries);
            Console.WriteLine(J);

            // Gradient descent
            int iterations = 1500;
            double learningRate = 0.01;
            GradientDescent(X, y, thetaSeries, learningRate, iterations);
        }

        private static void GradientDescent(Matrix<double> X, Vector<double> y, Series<int, double> thetaSeries, double learningRate, int iterations)
        {
            int numberOfExamples = X.RowCount;
            var CostHistory = Enumerable.Range(0, iterations).Select(i => KeyValue.Create(i, 0.0)).ToSeries();
            for (int i = 0; i < iterations; i++)
            {
                // Update theta
                // theta = theta - ((alpha/m) * sum(((X*theta)-y) .* X))';  

                // Save cost history
                // CostHistory(iter) = computeCost(X, y, theta);
            }
        }

        public static (Matrix<double> X, Vector<double> y) PrepareInputData(string csvPath)
        {
            // Load from csv
            var frame = Frame.ReadCsv(csvPath);

            // Prepare X Matrix
            Series<int, double> xSeries = frame.GetColumn<double>("X");
            Series<int, double> intercept = Enumerable.Range(0, xSeries.ValueCount).Select(i => KeyValue.Create(i, 1.0)).ToSeries();
            Frame<int, string> xFrame = Frame.CreateEmpty<int, string>();
            xFrame.AddColumn("X", xSeries);
            xFrame.AddColumn("Intercept", intercept);
            Matrix<double> X = Deedle.Math.Matrix.ofFrame(xFrame);

            // Preapare Y vector
            Series<int, double> ySeries = frame.GetColumn<double>("Y");
            Vector<double> y = Deedle.Math.Series.toVector(ySeries);

            return (X, y);
        }

        public static double ComputeCost(Matrix<double> X, Vector<double> y, Series<int, double> theta)
        {
            // J = sum(((X*theta)-y).^2) * (1/(2*m))
            var dotproduct = Deedle.Math.Matrix.dot(X, theta); // X*theta
            var dotproductMinusY = dotproduct - y; // (X*theta) - y
            var dotproductMinusYPower2 = dotproductMinusY.PointwisePower(2); // ((X*theta)-y).^2
            var dotproductMinusYPower2Sum = dotproductMinusYPower2.Sum(); // sum(((X*theta)-y).^2)
            var dotproductMinusYPower2SumDivided2M = dotproductMinusYPower2Sum * (1.0 / (2.0 * X.RowCount)); // sum(((X*theta)-y).^2)*(1/(2*m))

            return dotproductMinusYPower2SumDivided2M;
        }

        public static void PlotSeries(Series<int, double> x, Series<int, double> y)
        {
            var chart1 = Chart.Plot(
                new Scatter
                {
                    x = x.Values,
                    y = y.Values,
                    mode = "markers"
                });
            var chart1_layout = new Layout.Layout
            {
                title = "Predict profit for a food truck",
                xaxis = new Xaxis
                {
                    title = "Population of City in 10,000s"
                },
                yaxis = new Yaxis
                {
                    title = "Profit in $10,000s"
                }
            };
            chart1.WithLayout(chart1_layout);
            chart1.Show();
        }
    }
}
