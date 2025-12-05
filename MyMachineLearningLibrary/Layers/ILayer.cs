using MyMachineLearningLibrary.Activation_Functions;
using MyMachineLearningLibrary.Optimizers;
using MyMachineLearningLibrary.Perceptrons;
using System.Text.Json;

namespace MyMachineLearningLibrary.Layers
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<ILayer>))]
	public interface ILayer
	{
		public IPerceptron[] Perceptrons { get; set; }
		public IActivationFunction ActivationFunction { get; set; }
		public MatrixExtension Gradients { get; set; }
		public MatrixExtension Outputs { get; set; }
		public int NumberOfPerceptrons { get; set; }
		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, int layerIndex);
		public MatrixExtension FeedForward(MatrixExtension layerInputs);
		public MatrixExtension FeedForward(MatrixExtension layerInputs, bool x);
		public MatrixExtension TransposeWeights();
		public MatrixExtension Backpropagate(MatrixExtension errors, double learningRate, double decayRate, MatrixExtension previousLayerOutputs, int currentEpoch, IOptimizer optimizer);
	}
}
