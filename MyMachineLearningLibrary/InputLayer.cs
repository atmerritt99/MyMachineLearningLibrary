using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class InputLayer : ILayer
	{
		public IActivationFunction ActivationFunction { get; set; }
		public NeuralNetMatrix LayerOutputs { get; set; }
		public int NumberOfPerceptrons { get; set; }
		public NeuralNetMatrix Gradients { get; set; }
		[JsonIgnore]
		public NeuralNetMatrix Weights { get; set; }
		[JsonIgnore]
		public NeuralNetMatrix Biases { get; set; }

		public InputLayer(int NumberOfPerceptrons)
		{
			this.NumberOfPerceptrons = NumberOfPerceptrons;
			this.LayerOutputs = new NeuralNetMatrix(NumberOfPerceptrons, 1);
			this.Gradients = new NeuralNetMatrix(NumberOfPerceptrons, 1);
			ActivationFunction = new SigmoidActivationFunction();
		}
		
		public void ApplyGradients(NeuralNetMatrix previousLayer, int batchSize)
		{
			throw new NotImplementedException();
		}

		public void CalculateGradients(NeuralNetMatrix errors, double learningRate)
		{
			throw new NotImplementedException();
		}

		public NeuralNetMatrix FeedForward(NeuralNetMatrix neuralNetMatrix)
		{
			LayerOutputs = neuralNetMatrix;
			return LayerOutputs;
		}

		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, NeuralNetMatrix previousLayersGradients)
		{
			throw new NotImplementedException();
		}

		public NeuralNetMatrix TransposeWeights()
		{
			throw new NotImplementedException();
		}
	}
}
