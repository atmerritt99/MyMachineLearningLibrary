using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class SignActivationFunction : IActivationFunction
	{
		public int MaxClass { get; set; } = 1;
		public int MinClass { get; set; } = -1;

		public double ActivateDerivativeOfFunction(double x)
		{
			double e2 = Math.Exp(2 * x);
			double tanh = (e2 - 1) / (e2 + 1);
			return 1 - (tanh * tanh);
		}

		public NeuralNetMatrix ActivateDerivativeOfFunction(NeuralNetMatrix m)
		{
			NeuralNetMatrix result = new NeuralNetMatrix(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					double e2 = Math.Exp(2 * m[i, j]);
					double tanh = (e2 - 1) / (e2 + 1);
					result[i, j] = 1 - (tanh * tanh);
				}
			}
			return result;
		}

		public double ActivateFunction(double x)
		{
			return x >= 0 ? MaxClass : MinClass;
		}

		public NeuralNetMatrix ActivateFunction(NeuralNetMatrix m)
		{
			NeuralNetMatrix result = new NeuralNetMatrix(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					result[i, j] = m[i, j] >= 0 ? MaxClass : MinClass;
				}
			}
			return result;
		}
	}
}
