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
            var frame = Frame.ReadCsv("ex1data1.txt");
            var xSeries = frame.GetColumn<double>("X");
            var ySeries = frame.GetColumn<double>("Y");
            xSeries.Print();
            ySeries.Print();
            int numberOfExamples = ySeries.ValueCount;
            Console.WriteLine(numberOfExamples);
            //PlotSeries(xSeries, ySeries);

            var intercept = Enumerable.Range(0, numberOfExamples).Select(i => KeyValue.Create(i, 1.0)).ToSeries();
            List<Series<int, double>> xSeriesList = new List<Series<int, double>>()
            {
                xSeries,
                intercept
            };
            var xFrame = Frame.CreateEmpty<int, string>();
            xFrame.AddColumn("X", xSeries);
            xFrame.AddColumn("Intercept", intercept);
            xFrame.Print();
            var thetaSeries = new Series<int, double>(new int[2] { 0, 1 }, new double[2]);
            int iterations = 1500;
            double learningRate = 0.01;

            Vector<double> cost = CreateVector.Dense<double>(numberOfExamples, i => 0);
            Vector<double> multiplier = CreateVector.Dense<double>(numberOfExamples, i => (1 / (2 * numberOfExamples)));

            var X = Deedle.Math.Matrix.ofFrame(xFrame);
            var y = Deedle.Math.Series.toVector(ySeries);
            Console.WriteLine(X);
            Console.WriteLine(y);

            // sum(((X*theta)-y).^2) * (1/(2*m));
            var dotproduct = Deedle.Math.Matrix.dot(X, thetaSeries);
            var dotproductMinusY = dotproduct - y;
            var dotproductMinusYPower2 = dotproductMinusY.PointwisePower(2);
            var dotproductMinusYPower2Sum = dotproductMinusYPower2.Sum();
            var dotproductMinusYPower2SumDivided2M = dotproductMinusYPower2Sum * (1.0 / (2.0 * numberOfExamples));
            var J = dotproductMinusYPower2SumDivided2M;
            Console.WriteLine(J);
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
