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
        public int ElementCount
        { 
            get
            {
                return RowLength * ColoumnLength;
            }
        }

        public double Sum
        {
            get
            {
                double sum = 0;
                foreach (var row in Values)
                {
                    sum += row.Sum();
                }
                return sum;
            }
        }

		public double Average
		{
			get
			{
                return Sum / ElementCount;
			}
		}

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

        public double Max
        {
            get
            {
                return Flatten().Max();
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

        public void ScalarDivide(double x)
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
		public void Divide(NeuralNetMatrix m)
		{
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
                    Values[i][j] /= m[i, j];
				}
			}
		}

		public static NeuralNetMatrix ScalarMultiply(NeuralNetMatrix a, double x)
		{
            var result = new NeuralNetMatrix(a.RowLength, a.ColoumnLength);

			for (int i = 0; i < a.RowLength; i++)
			{
				for (int j = 0; j < a.ColoumnLength; j++)
				{
					result[i, j] *= x;
				}
			}

            return result;
		}

		/// <summary>
		/// Performs an element wise multiplication of two matrices
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		public static NeuralNetMatrix Multiply(NeuralNetMatrix a, NeuralNetMatrix b)
		{
			var result = new NeuralNetMatrix(a.RowLength, a.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = a[i, j] * b[i, j];
				}
			}
			return result;
		}

		public static NeuralNetMatrix ScalarSubtract(NeuralNetMatrix a, double b)
		{
			var result = new NeuralNetMatrix(a.RowLength, a.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = a[i, j] - b;
				}
			}
			return result;
		}

		public static NeuralNetMatrix ScalarSubtract(double a, NeuralNetMatrix b)
		{
			var result = new NeuralNetMatrix(b.RowLength, b.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = a - b[i, j];
				}
			}
			return result;
		}

		public static NeuralNetMatrix Subtract(NeuralNetMatrix a, NeuralNetMatrix b)
		{
            var result = new NeuralNetMatrix(a.RowLength, a.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
                    result[i,j] = a[i, j] - b[i, j];
				}
			}
            return result;
		}

		public static NeuralNetMatrix ScalarAbs(NeuralNetMatrix a)
		{
			var result = new NeuralNetMatrix(a.RowLength, a.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
                    result[i, j] = Math.Abs(a[i, j]);
				}
			}
			return result;
		}

		public static NeuralNetMatrix Log(NeuralNetMatrix a)
		{
			var result = new NeuralNetMatrix(a.RowLength, a.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = Math.Log(a[i, j]);
				}
			}
			return result;
		}

		public static NeuralNetMatrix Exponent(NeuralNetMatrix a)
		{
			var result = new NeuralNetMatrix(a.RowLength, a.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = Math.Exp(a[i, j]);
				}
			}
			return result;
		}

        public double RowSum(int rowIndex)
        {
            return Values[rowIndex].Sum();
        }

		//public double RowMax(int rowIndex)
		//{
		//	return Values[rowIndex].Max();
		//}

		/// <summary>
		/// Compares two NeuralNetMatrices to see if one has larger values than the other
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns>A NeuralNetMatrix with 1 where a's value is bigger than b, -1 where b is bigger than a, and 0 where they are the same</returns>
		public static NeuralNetMatrix Compare(NeuralNetMatrix a, NeuralNetMatrix b)
        {
			var result = new NeuralNetMatrix(a.RowLength, a.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
                    if (a[i, j] < b[i, j])
                    {
                        result[i, j] = -1;
                    }
                    else if (a[i, j] > b[i, j])
					{
						result[i, j] = 1;
					}
                    else
                    {
						result[i, j] = 0;
					}
				}
			}
			return result;
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

        public NeuralNetMatrix Transpose()
        {
			var transposition = new NeuralNetMatrix(ColoumnLength, RowLength);

			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					transposition[j, i] = this[i, j];
				}
			}

			return transposition;
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

        public NeuralNetMatrix Diagonal()
        {
            var result = new NeuralNetMatrix(RowLength, RowLength);

            for(int i = 0; i < RowLength; i++)
            {
                result[i, i] = this[i, 0];
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
                    Values[i][j] = (rng.NextDouble() * 2) - 1;
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
                    Values[i][j] = rng.NextDouble() < mutationRate ? GaussianDistribution(0, .1) : Values[i][j];
                }
            }
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
