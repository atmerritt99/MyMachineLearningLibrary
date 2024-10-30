using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyMachineLearningLibrary
{
    public struct NeuralNetMatrix
    {
        public double[][] Values { get; set; }
        public  int RowLength { get; set; }
        public int ColoumnLength { get; set; }

        public double this[int r, int c]
        {
            get
            {
                return Values[r][c];
            }

            set
            {
                Values[r][c] = value;
            }
        }

        public NeuralNetMatrix()
        {
            Values = [];
            RowLength = 0;
            ColoumnLength = 0;
        }

        public NeuralNetMatrix(int rowLength, int coloumnLength)
        {
            Values = new double[rowLength][];

            for(int rowIndex = 0; rowIndex < Values.Length; rowIndex++)
                Values[rowIndex] = new double[coloumnLength];

            RowLength = rowLength;
            ColoumnLength = coloumnLength;
        }

        public NeuralNetMatrix(double[] values)
        {
            Values = new double[values.Length][];
            RowLength = values.Length;
            ColoumnLength = 1;

            for (int i = 0; i < RowLength; i++)
            {
                Values[i] = new double[1];
                Values[i][0] = values[i];
            }
        }

        public void ScalarMultiply(double x)
        {
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    Values[i][j] *= x; 
                }
            }
        }

        public void ScalarDivide(int x)
        {
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    Values[i][j] /= x;
                }
            }
        }

        public void ScalarAbs()
        {
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    Values[i][j] = Math.Abs(Values[i][j]);
                }
            }
        }

        public void Add(NeuralNetMatrix m)
        {
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    Values[i][j] += m[i, j]; 
                }
            }
        }

        public void Subtract(NeuralNetMatrix m)
        {
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    Values[i][j] -= m[i, j]; 
                }
            }
        }

        public void Multiply(NeuralNetMatrix m)
        {
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    Values[i][j] *= m[i, j];
                }
            }
        }

        public static NeuralNetMatrix DotProduct(NeuralNetMatrix a, NeuralNetMatrix b)
        {
            if (a.ColoumnLength != b.RowLength)
                throw new Exception("Columns and Rows must equal");

            NeuralNetMatrix result = new NeuralNetMatrix(a.RowLength, b.ColoumnLength);

            for (int i = 0; i < result.RowLength; i++)
            {
                for (int j = 0; j < result.ColoumnLength; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < a.ColoumnLength; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                }
            }

            return result;
        }

        public static NeuralNetMatrix Transpose(NeuralNetMatrix a)
        {
            NeuralNetMatrix result = new NeuralNetMatrix(a.ColoumnLength, a.RowLength);

            for (int i = 0; i < a.RowLength; i++)
            {
                for (int j = 0; j < a.ColoumnLength; j++)
                {
                    result[j, i] = a[i,j];
                }
            }

            return result;
        }

        public double[] Flatten()
        {
            double[] result = new double[RowLength * ColoumnLength];
            int counter = 0;

            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    result[counter] = Values[i][j];
                    counter++;
                }
            }

            return result;
        }

        public void Randomize()
        {
            Random rng = new();
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    Values[i][j] = (double)(rng.NextDouble() * 2) - 1;
                }
            }
        }

        public void Mutate(double mutationRate)
        {
            Random rng = new();
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    Values[i][j] = rng.NextDouble() < mutationRate ? (double)GaussianDistribution(0, .1) : Values[i][j];
                }
            }
        }

        public override string ToString()
        {
            string result = "";

            for(int i = 0; i < RowLength; i++)
            {
                for(int j = 0; j < ColoumnLength - 1; j++)
                {
                    result += $"{Values[i][j]},";
                }
                result += $"{Values[i][ColoumnLength - 1]}\n";
            }

            return result;
        }

        public NeuralNetMatrix Copy()
        {
            var copy = new NeuralNetMatrix(RowLength, ColoumnLength);
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    copy[i, j] = Values[i][j];
                }
            }
            return copy;
        }

        private double GaussianDistribution(double mean, double stddev)
        {
            Random rand = new Random(); //reuse this if you are generating many
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + stddev * randStdNormal; //random normal(mean,stdDev^2)
            return randNormal;
        }
    }
}
