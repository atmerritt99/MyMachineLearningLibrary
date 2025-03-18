using MyMachineLearningLibrary.Activation_Functions;
using MyMachineLearningLibrary.Optimizers;
using MyMachineLearningLibrary.Perceptrons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Layers
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<ILayer>))]
	public interface ILayer
	{
		public IPerceptron[] Perceptrons { get; set; }
		public IActivationFunction ActivationFunction { get; set; }
		public MatrixExtension Gradients { get; set; }
		public MatrixExtension LayerOutputs { get; set; }
		public int NumberOfPerceptrons { get; set; }
		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, int layerIndex);
		public MatrixExtension FeedForward(MatrixExtension layerInputs);
		public MatrixExtension TransposeWeights();
		public MatrixExtension Backpropagate(MatrixExtension errors, double learningRate, double decayRate, MatrixExtension previousLayer, int batchSize, int currentEpoch, IOptimizer optimizer);
	}
}
