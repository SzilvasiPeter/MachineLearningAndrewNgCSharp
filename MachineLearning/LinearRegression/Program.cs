using System;
using XPlot.Plotly;
using System.Linq;
using Deedle;

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
            var chart1 = Chart.Plot(
                new Scatter
                {
                    x = xSeries.Values,
                    y = ySeries.Values,
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
