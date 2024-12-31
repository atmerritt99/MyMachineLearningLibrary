using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class SoftmaxActivationFunction : IActivationFunction
	{
		public NeuralNetMatrix ActivateDerivativeOfFunction(NeuralNetMatrix m)
		{
			//var result = m.Diagonal();

			//for (int i = 0; i < m.ColoumnLength; i++)
			//{
			//	for (int j = 0; j < m.ColoumnLength; j++)
			//	{
			//		if(i == j)
			//		{
			//			result[i, j] = -m[0, i] * m[0, j];
			//			continue;
			//		}

			//		result[i, j] = m[0, i] * (1 - m[0, i]);
			//	}
			//}

			//return result;

			throw new NotImplementedException();
		}

		public NeuralNetMatrix ActivateFunction(NeuralNetMatrix m)
		{
			var result = new NeuralNetMatrix(m.RowLength, m.ColoumnLength);

			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					result[i, j] = m[i, j] - m.RowMax(i);
				}
			}

			result = NeuralNetMatrix.Exponent(result);

			for (int i = 0; i < m.RowLength; i++)
			{
				for(int j = 0; j < m.ColoumnLength; j++)
				{
					result[i, j] = result[i, j] / result.RowSum(i);
				}
			}

			return result;
		}
	}
}
