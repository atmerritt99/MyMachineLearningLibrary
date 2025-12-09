using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyMachineLearningLibrary
{
    public class MatrixExtension
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

		public double RowSumOfAvg
		{
			get
			{
				double sum = 0;
				foreach (var row in Values)
				{
					sum += row.Average();
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

		public double[] this[int r]
		{
			get
			{
				return Values[r];
			}

			set
			{
				Values[r] = value;
			}
		}

		public double Max
        {
            get
            {
                return Flatten().Max();
            }
        }

        public MatrixExtension()
        {
            Values = [];
            RowLength = 0;
            ColoumnLength = 0;
        }

        public MatrixExtension(int rowLength, int coloumnLength)
        {
            Values = new double[rowLength][];

            for(int rowIndex = 0; rowIndex < Values.Length; rowIndex++)
                Values[rowIndex] = new double[coloumnLength];

            RowLength = rowLength;
            ColoumnLength = coloumnLength;
        }

		public MatrixExtension(double[] values)
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

		public MatrixExtension(double[][] values)
		{
			Values = new double[values.Length][];
			RowLength = values.Length;
			ColoumnLength = values[0].Length;

			for (int i = 0; i < RowLength; i++)
			{
				Values[i] = values[i];
			}
		}

		public MatrixExtension Multiply(double x)
        {
            var result = new MatrixExtension(RowLength, ColoumnLength);
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    result[i, j] = Values[i][j] * x; 
                }
            }
            return result;
        }

        public MatrixExtension Divide(double x)
        {
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] / x;
				}
			}
			return result;
		}

        public MatrixExtension AbsoluteValue()
        {
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Math.Abs(Values[i][j]);
				}
			}
			return result;
		}

        public MatrixExtension Add(MatrixExtension m)
        {
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
                    result[i, j] = Values[i][j] + m[i, j];
				}
			}
			return result;
		}

		public MatrixExtension AddToEachRow(MatrixExtension m)
		{
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] + m[i, 0];
				}
			}
			return result;
		}

		public MatrixExtension Add(double m)
		{
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] + m;
				}
			}
			return result;
		}

		public MatrixExtension Subtract(double m)
		{
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] - m;
				}
			}
			return result;
		}

		public MatrixExtension Subtract(MatrixExtension m)
		{
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] - m[i, j];
				}
			}
			return result;
		}

		public MatrixExtension Multiply(MatrixExtension m)
        {
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] * m[i, j];
				}
			}
			return result;
		}
		public MatrixExtension Divide(MatrixExtension m)
		{
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < RowLength; i++)
			{
				for (int j = 0; j < ColoumnLength; j++)
				{
					result[i, j] = Values[i][j] / m[i, j];
				}
			}
			return result;
		}

		public static MatrixExtension Subtract(double a, MatrixExtension b)
		{
			var result = new MatrixExtension(b.RowLength, b.ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = a - b[i, j];
				}
			}
			return result;
		}

		public MatrixExtension Log()
		{
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = Math.Log(Values[i][j]);
				}
			}
			return result;
		}

		public MatrixExtension Exponent()
		{
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = Math.Exp(Values[i][j]);
				}
			}
			return result;
		}

		public MatrixExtension SquareRoot()
		{
			var result = new MatrixExtension(RowLength, ColoumnLength);
			for (int i = 0; i < result.RowLength; i++)
			{
				for (int j = 0; j < result.ColoumnLength; j++)
				{
					result[i, j] = Math.Sqrt(Values[i][j]);
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
		public static MatrixExtension Compare(MatrixExtension a, MatrixExtension b)
        {
			var result = new MatrixExtension(a.RowLength, a.ColoumnLength);
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

		public MatrixExtension DotProduct(MatrixExtension b)
		{
			if (ColoumnLength != b.RowLength)
				throw new Exception("Columns and Rows must equal");

			MatrixExtension result = new MatrixExtension(RowLength, b.ColoumnLength);

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

		public MatrixExtension Transpose()
        {
			var transposition = new MatrixExtension(ColoumnLength, RowLength);

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

		public MatrixExtension Diagonal()
        {
            var result = new MatrixExtension(RowLength, RowLength);

            for(int i = 0; i < RowLength; i++)
            {
                result[i, i] = this[i, 0];
            }

            return result;
        }

        public MatrixExtension Copy()
        {
            var copy = new MatrixExtension(RowLength, ColoumnLength);
            for (int i = 0; i < RowLength; i++)
            {
                for (int j = 0; j < ColoumnLength; j++)
                {
                    copy[i, j] = Values[i][j];
                }
            }
            return copy;
        }

		public MatrixExtension GetRows(int start, int numberOfRows)
		{
			var result = new MatrixExtension(numberOfRows, ColoumnLength);
			for(int i = start; i < start + numberOfRows; i++)
			{
				result.Values[i - start] = Values[i];
			}
			return result;
		}
	}
}
