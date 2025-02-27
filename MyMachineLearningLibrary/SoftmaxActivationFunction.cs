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
			var result = m.Subtract(m.Max);
			result = result.Exponent();
			result = result.Divide(result.Sum);
			return result;
		}

		public double ActivateFunction(double x)
		{
			throw new NotImplementedException();
		}
	}
}
