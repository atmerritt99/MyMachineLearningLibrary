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

		public NeuralNetMatrix Multiply(double x)
        {
            var result = new NeuralNetMatrix(RowLength, ColoumnLength);
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    result[i, j] = Values[i][j] * x; 
                }
            }
            return result;
        }

        public NeuralNetMatrix Divide(double x)
        {
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] / x;
				}
			}
			return result;
		}

        public NeuralNetMatrix AbsoluteValue()
        {
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Math.Abs(Values[i][j]);
				}
			}
			return result;
		}

        public NeuralNetMatrix Add(NeuralNetMatrix m)
        {
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
                    result[i, j] = Values[i][j] + m[i, j];
				}
			}
			return result;
		}

		public NeuralNetMatrix Subtract(double m)
		{
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] - m;
				}
			}
			return result;
		}

		public NeuralNetMatrix Subtract(NeuralNetMatrix m)
		{
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] - m[i, j];
				}
			}
			return result;
		}

		public NeuralNetMatrix Multiply(NeuralNetMatrix m)
        {
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] * m[i, j];
				}
			}
			return result;
		}
		public NeuralNetMatrix Divide(NeuralNetMatrix m)
		{
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] / m[i, j];
				}
			}
			return result;
		}

		public static NeuralNetMatrix Subtract(double a, NeuralNetMatrix b)
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

		public NeuralNetMatrix Log()
		{
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = Math.Log(Values[i][j]);
				}
			}
			return result;
		}

		public NeuralNetMatrix Exponent()
		{
			var result = new NeuralNetMatrix(RowLength, ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = Math.Exp(Values[i][j]);
				}
			}
			return result;
		}

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

		public NeuralNetMatrix DotProduct(NeuralNetMatrix b)
		{
			if (ColoumnLength != b.RowLength)
				throw new Exception("Columns and Rows must equal");

			NeuralNetMatrix result = new NeuralNetMatrix(RowLength, b.ColoumnLength);

			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					double sum = 0;
					for (int k = 0; k < ColoumnLength; k++)
					{
						sum += Values[i][k] * b[k, j];
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
    }
}
