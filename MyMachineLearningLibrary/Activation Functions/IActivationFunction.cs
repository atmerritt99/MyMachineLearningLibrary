using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Activation_Functions
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<IActivationFunction>))]
	public interface IActivationFunction
	{
		public int MaxClass { get; set; }
		public int MinClass { get; set; }
		public double ActivateFunction(double x);
		public double ActivateDerivativeOfFunction(double x);
		public MatrixExtension ActivateFunction(MatrixExtension m);
		// With the Derivative Assume the input as gone through the activation function
		public MatrixExtension ActivateDerivativeOfFunction(MatrixExtension m);
	}
}
