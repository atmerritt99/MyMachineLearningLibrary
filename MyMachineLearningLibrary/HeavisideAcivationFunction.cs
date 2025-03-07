using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class HeavisideAcivationFunction : IActivationFunction
	{
		public double Threshold { get; set; }
		public int MaxClass { get; set; } = 1;
		public int MinClass { get; set; } = 0;

		public HeavisideAcivationFunction(double threshold = 0)
		{
			Threshold = threshold;
		}

		/// <summary>
		/// Returns the derivative of sigmoid because the actual derivative of Heaviside is problematic for the purposes of machine learning. Sigmoid is an adaquete approximation.
		/// </summary>
		/// <param name="m"></param>
		/// <returns></returns>
		public NeuralNetMatrix ActivateDerivativeOfFunction(NeuralNetMatrix m)
		{
			NeuralNetMatrix result = new NeuralNetMatrix(m.RowLength, m.ColoumnLength);
			for (int i = 0; i < m.RowLength; i++)
			{
				for (int j = 0; j < m.ColoumnLength; j++)
				{
					double sigmoid = 1.0 / (1.0 + Math.Exp(-m[i, j]));
					result[i, j] = sigmoid * (1 - sigmoid);
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
					result[i, j] = m[i, j] >= Threshold ? MaxClass : MinClass;
				}
			}
			return result;
		}

		public double ActivateFunction(double x)
		{
			return x >= Threshold ? MaxClass : MinClass;
		}

		/// <summary>
		/// Returns the derivative of sigmoid because the actual derivative of Heaviside is problematic for the purposes of machine learning. Sigmoid is an adaquete approximation.
		/// </summary>
		/// <param name="m"></param>
		/// <returns></returns>
		public double ActivateDerivativeOfFunction(double x)
		{
			double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
			return sigmoid * (1 - sigmoid);
		}
	}
}
