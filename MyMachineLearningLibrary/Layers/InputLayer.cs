using MyMachineLearningLibrary.Activation_Functions;
using MyMachineLearningLibrary.Optimizers;
using MyMachineLearningLibrary.Perceptrons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Layers
{
	public class InputLayer : ILayer
	{
		public IActivationFunction ActivationFunction { get; set; }
		public MatrixExtension LayerOutputs { get; set; }
		public int NumberOfPerceptrons { get; set; }
		public MatrixExtension Gradients { get; set; }
		[JsonIgnore]
		public MatrixExtension Weights { get; set; }
		[JsonIgnore]
		public MatrixExtension Biases { get; set; }
		public IPerceptron[] Perceptrons { get; set; }

		public InputLayer(int NumberOfPerceptrons)
		{
			this.NumberOfPerceptrons = NumberOfPerceptrons;
			this.LayerOutputs = new MatrixExtension(NumberOfPerceptrons, 1);
			this.Gradients = new MatrixExtension(NumberOfPerceptrons, 1);
			ActivationFunction = new SigmoidActivationFunction();
		}

		public void ApplyGradients(MatrixExtension previousLayer, int batchSize)
		{
			throw new NotImplementedException();
		}

		public void CalculateGradients(MatrixExtension errors, double learningRate)
		{
			throw new NotImplementedException();
		}

		public MatrixExtension FeedForward(MatrixExtension neuralNetMatrix)
		{
			LayerOutputs = neuralNetMatrix;
			return LayerOutputs;
		}

		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, ILayer layer, int layerIndex)
		{
			throw new NotImplementedException();
		}

		public MatrixExtension TransposeWeights()
		{
			throw new NotImplementedException();
		}

		public MatrixExtension Backpropagate(MatrixExtension errors, double learningRate, double decayRate, MatrixExtension previousLayer, int batchSize, int currentEpoch, IOptimizer optimizer)
		{
			throw new NotImplementedException();
		}
	}
}
