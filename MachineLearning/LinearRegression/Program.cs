using System;
using XPlot.Plotly; // Using OxyPlot WPF application instead of Plotly
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
            double cost = ComputeCost(X, y, thetaSeries);
            Console.WriteLine(cost);

            // Gradient descent
            int iterations = 1500;
            double learningRate = 0.01;
            double[] costHistory = GradientDescent(X, y, ref thetaSeries, learningRate, iterations);
            foreach (var aCost in costHistory)
            {
                //Console.WriteLine(aCost);
            }

            //Console.WriteLine(thetaSeries);

            // Visualizing Data and regression line
            double[] xArray = X.Column(1).ToArray();
            double[] yArray = y.ToArray();
            double[] thetaArray = Deedle.Math.Series.toVector(thetaSeries).ToArray();

            PlotSeries(xArray, yArray, thetaArray);
        }

        private static double[] GradientDescent(Matrix<double> X, Vector<double> y, ref Series<int, double> thetaSeries, double learningRate, int iterations)
        {
            int numberOfExamples = X.RowCount;
            double[] costHistory = new double[iterations];
            for (int i = 0; i < iterations; i++)
            {
                // Update theta
                // theta = theta - ((alpha/m) * sum(((X*theta)-y) .* X))';
                var dotProduct = Deedle.Math.Matrix.dot(X, thetaSeries); // X*theta
                var dotProductMinusY = (dotProduct - y).ToRowMatrix(); // (X*theta)-y
                var dotProductMinusYMultiplyX = dotProductMinusY.Multiply(X); // ((X*theta)-y) .* X)
                var dotProductMinusYMultiplyXSum = dotProductMinusYMultiplyX.ColumnSums(); // sum(((X*theta)-y) .* X))
                var dotProductMinusYMultiplyXSumLr = dotProductMinusYMultiplyXSum.Multiply(learningRate / numberOfExamples); // (alpha/m) * sum(((X*theta)-y) .* X)

                var thetaVector = Deedle.Math.Series.toVector(thetaSeries);
                var newThetaVector = thetaVector - dotProductMinusYMultiplyXSumLr; // theta - ((alpha/m) * sum(((X*theta)-y) .* X))
                thetaSeries = Deedle.Math.Series.ofVector(Enumerable.Range(0, dotProductMinusYMultiplyXSumLr.Count), newThetaVector);

                // Save cost history
                costHistory[i] = ComputeCost(X, y, thetaSeries);
            }

            return costHistory;
        }

        public static (Matrix<double> X, Vector<double> y) PrepareInputData(string csvPath)
        {
            // Load from csv
            var frame = Frame.ReadCsv(csvPath);

            // Prepare X Matrix
            Series<int, double> xSeries = frame.GetColumn<double>("X");
            Series<int, double> intercept = Enumerable.Range(0, xSeries.ValueCount).Select(i => KeyValue.Create(i, 1.0)).ToSeries();
            Frame<int, string> xFrame = Frame.CreateEmpty<int, string>();
            xFrame.AddColumn("Intercept", intercept);
            xFrame.AddColumn("X", xSeries);
            Matrix<double> X = Deedle.Math.Matrix.ofFrame(xFrame);

            // Preapare Y vector
            Series<int, double> ySeries = frame.GetColumn<double>("Y");
            Vector<double> y = Deedle.Math.Series.toVector(ySeries);

            return (X, y);
        }

        public static double ComputeCost(Matrix<double> X, Vector<double> y, Series<int, double> theta)
        {
            // sum(((X*theta)-y).^2) * (1/(2*m))
            var dotproduct = Deedle.Math.Matrix.dot(X, theta); // X*theta
            var dotproductMinusY = dotproduct - y; // (X*theta) - y
            var dotproductMinusYPower2 = dotproductMinusY.PointwisePower(2); // ((X*theta)-y).^2
            var dotproductMinusYPower2Sum = dotproductMinusYPower2.Sum(); // sum(((X*theta)-y).^2))
            var dotproductMinusYPower2SumDivided2M = dotproductMinusYPower2Sum * (1.0 / (2.0 * X.RowCount)); // sum(((X*theta)-y).^2)*(1/(2*m))

            return dotproductMinusYPower2SumDivided2M;
        }

        public static void PlotSeries(double[] x, double[] y, double[] theta)
        {
            double yIntercept = theta[0];
            double slope = theta[1];
            var chart_list = new List<Scatter>
            {
                new Scatter
                {
                    x = x,
                    y = y,
                    mode = "markers",
                    name = "Data points"
                },
                new Scatter
                {
                    // Draw a line in [3, 24] intervallum given theta parameters
                    x = new double[] { 3, 24 },
                    y = new double[] { 3 * slope + yIntercept,  24 * slope + yIntercept },
                    mode= "lines",
                    name = "Linear regression line"
                }
            };
            var chart = Chart.Plot(chart_list);

            var chart_layout = new Layout.Layout
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

            chart.WithLayout(chart_layout);
            chart.Show();
        }
    }
}
