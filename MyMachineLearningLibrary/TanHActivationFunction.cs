using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class TanHActivationFunction : IActivationFunction
	{
		public int MaxClass { get; set; } = 1;
		public int MinClass { get; set; } = -1;

		public NeuralNetMatrix ActivateDerivativeOfFunction(NeuralNetMatrix m)
		{
			NeuralNetMatrix result = new NeuralNetMatrix(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					result[i, j] = 1 - (m[i, j] * m[i, j]);
				}
			}
			return result;
		}

		public double ActivateDerivativeOfFunction(double x)
		{
			return 1 - (x * x);
		}

		public NeuralNetMatrix ActivateFunction(NeuralNetMatrix m)
		{
			NeuralNetMatrix result = new NeuralNetMatrix(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					double e2 = Math.Exp(2 * m[i, j]);
					result[i, j] = (e2 - 1) / (e2 + 1);
				}
			}
			return result;
		}

		public double ActivateFunction(double x)
		{
			double e2 = Math.Exp(2 * x);
			return (e2 - 1) / (e2 + 1);
		}
	}
}
