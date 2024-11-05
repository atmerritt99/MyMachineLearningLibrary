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
		public IActivationFunction ActivationFunction { get; set; }
		public NeuralNetMatrix Gradients { get; set; }
		public NeuralNetMatrix Weights { get; set; }
		public NeuralNetMatrix Biases { get; set; }
		public NeuralNetMatrix LayerOutputs { get; set; }
		public int NumberOfPerceptrons { get; set; }
		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, NeuralNetMatrix previousLayersGradients);
		public NeuralNetMatrix FeedForward(NeuralNetMatrix neuralNetMatrix);
		public void CalculateGradients(NeuralNetMatrix errors, double learningRate);
		public void ApplyGradients(NeuralNetMatrix previousLayer, int batchSize);
	}
}
