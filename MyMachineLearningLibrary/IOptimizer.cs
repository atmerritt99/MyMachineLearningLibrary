using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<IOptimizer>))]
	public interface IOptimizer
	{
		void Compile(NeuralNetwork neuralNetwork);
		NeuralNetMatrix OptimizeGradients(NeuralNetMatrix gradients, double learningRate, double decayRate, int currentEpoch, int layerIndex);
	}
}
