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
			throw new NotImplementedException();
		}

		public double ActivateDerivativeOfFunction(double x)
		{
			throw new NotImplementedException();
		}

		public NeuralNetMatrix ActivateFunction(NeuralNetMatrix m)
		{
			var result = NeuralNetMatrix.ScalarSubtract(m, m.Max);
			result = NeuralNetMatrix.Exponent(result);
			result.ScalarDivide(result.Sum);
			return result;
		}

		public double ActivateFunction(double x)
		{
			throw new NotImplementedException();
		}
	}
}
