using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class TanHActivationFunction : IActivationFunction
	{
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

		public NeuralNetMatrix ActivateFunction(NeuralNetMatrix m)
		{
			NeuralNetMatrix result = new NeuralNetMatrix(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					double e2 = (double)Math.Exp((double)(2 * m[i, j]));
					result[i, j] = (e2 - 1) / (e2 + 1);
				}
			}
			return result;
		}
	}
}
