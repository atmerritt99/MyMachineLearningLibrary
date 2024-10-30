using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<IActivationFunction>))]
	public interface IActivationFunction
	{
		public NeuralNetMatrix ActivateFunction(NeuralNetMatrix m);
		// With the Derivative Assume the input as gone through the activation function
		public NeuralNetMatrix ActivateDerivativeOfFunction(NeuralNetMatrix m);
	}
}
