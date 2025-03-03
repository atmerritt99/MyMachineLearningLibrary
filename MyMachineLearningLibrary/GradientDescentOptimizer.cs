using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class GradientDescentOptimizer : IOptimizer
	{
		public void Compile(NeuralNetwork neuralNetwork)
		{
			throw new NotImplementedException();
		}

		public NeuralNetMatrix OptimizeGradients(NeuralNetMatrix gradients, double learningRate, double decayRate, int currentEpoch, int layerIndex)
		{
			//Calculate Decay
			var decay = currentEpoch / decayRate;
			var alpha = decayRate == 0 ? learningRate : learningRate / (Math.Sqrt(decay) + 1e-8);
			return gradients.Multiply(learningRate);
		}
	}
}
