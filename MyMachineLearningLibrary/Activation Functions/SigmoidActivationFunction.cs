﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Activation_Functions
{
	public class SigmoidActivationFunction : IActivationFunction
	{
		public int MaxClass { get; set; } = 1;
		public int MinClass { get; set; } = 0;

		public MatrixExtension ActivateDerivativeOfFunction(MatrixExtension m)
		{
			MatrixExtension result = new MatrixExtension(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					result[i, j] = m[i, j] * (1 - m[i, j]);
				}
			}
			return result;
		}

		public double ActivateDerivativeOfFunction(double x)
		{
			return x * (1 - x);
		}

		public MatrixExtension ActivateFunction(MatrixExtension m)
		{
			MatrixExtension result = new MatrixExtension(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					result[i, j] = 1.0 / (1.0 + Math.Exp(-m[i, j]));
				}
			}
			return result;
		}

		public double ActivateFunction(double x)
		{
			return 1.0 / (1.0 + Math.Exp(-x));
		}
	}
}
