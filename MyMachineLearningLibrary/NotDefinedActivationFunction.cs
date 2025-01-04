using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class NotDefinedActivationFunction : IActivationFunction
	{
		public double ActivateDerivativeOfFunction(double x)
		{
			throw new NotImplementedException();
		}

		public NeuralNetMatrix ActivateDerivativeOfFunction(NeuralNetMatrix m)
		{
			throw new NotImplementedException();
		}

		public double ActivateFunction(double x)
		{
			throw new NotImplementedException();
		}

		public NeuralNetMatrix ActivateFunction(NeuralNetMatrix m)
		{
			throw new NotImplementedException();
		}
	}
}
