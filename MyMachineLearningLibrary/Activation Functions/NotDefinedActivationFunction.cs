using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Activation_Functions
{
	public class NotDefinedActivationFunction : IActivationFunction
	{
		public int MaxClass { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
		public int MinClass { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

		public double ActivateDerivativeOfFunction(double x)
		{
			throw new NotImplementedException();
		}

		public MatrixExtension ActivateDerivativeOfFunction(MatrixExtension m)
		{
			throw new NotImplementedException();
		}

		public double ActivateFunction(double x)
		{
			throw new NotImplementedException();
		}

		public MatrixExtension ActivateFunction(MatrixExtension m)
		{
			throw new NotImplementedException();
		}
	}
}
