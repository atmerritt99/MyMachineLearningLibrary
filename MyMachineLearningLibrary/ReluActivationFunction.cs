using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class ReluActivationFunction : IActivationFunction
	{
		public double Leak { get; set; }

		public ReluActivationFunction(double Leak = 0)
		{
			if (Leak >= 1 || Leak < 0)
				throw new Exception("The leak value should be a double greater than or equal to 0 and less than 1");

			this.Leak = Leak;
		}

		public NeuralNetMatrix ActivateDerivativeOfFunction(NeuralNetMatrix m)
		{
			NeuralNetMatrix result = new NeuralNetMatrix(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					result[i, j] = m[i, j] >= 0 ? 1 : Leak;
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
					result[i, j] = Math.Max(Leak * m[i, j], m[i, j]);
				}
			}
			return result;
		}
	}
}
