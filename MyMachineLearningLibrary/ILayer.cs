using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<ILayer>))]
	public interface ILayer
	{
		public IPerceptron[] Perceptrons { get; set; }
		public IActivationFunction ActivationFunction { get; set; }
		public NeuralNetMatrix Gradients { get; set; }
		public NeuralNetMatrix LayerOutputs { get; set; }
		public int NumberOfPerceptrons { get; set; }
		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, ILayer previousLayer, int layerIndex);
		public NeuralNetMatrix FeedForward(NeuralNetMatrix neuralNetMatrix);
		public NeuralNetMatrix TransposeWeights();
		public NeuralNetMatrix Backpropagate(NeuralNetMatrix errors, double learningRate, double decayRate, NeuralNetMatrix previousLayer, int batchSize, int currentEpoch, IOptimizer optimizer);
	}
}
