using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class DenseLayer : ILayer
	{
		public IActivationFunction ActivationFunction { get; set; }
		public NeuralNetMatrix LayerOutputs { get; set; }
		public NeuralNetMatrix Weights { get; set; }
		public NeuralNetMatrix Biases { get; set; }
		public int NumberOfPerceptrons { get; set; }
		public NeuralNetMatrix Gradients { get; set; }

		public DenseLayer(int numberOfPerceptrons, IActivationFunction activationFunction) 
		{
			Biases = new NeuralNetMatrix(numberOfPerceptrons, 1);
			Biases.Randomize();
			NumberOfPerceptrons = numberOfPerceptrons;
			ActivationFunction = activationFunction;
		}

		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, NeuralNetMatrix previousLayersGradients)
		{
			Weights = new NeuralNetMatrix(NumberOfPerceptrons, numberOfPerceptronsInPreviousLayer);
			Weights.Randomize();

			Gradients = NeuralNetMatrix.DotProduct(Weights, previousLayersGradients);
		}

		public NeuralNetMatrix FeedForward(NeuralNetMatrix neuralNetMatrix)
		{
			LayerOutputs = NeuralNetMatrix.DotProduct(Weights, neuralNetMatrix);
			LayerOutputs.Add(Biases);
			LayerOutputs = ActivationFunction.ActivateFunction(LayerOutputs);
			return LayerOutputs;
		}

		public void CalculateGradients(NeuralNetMatrix errors, double learningRate)
		{
			var outputGradients = ActivationFunction.ActivateDerivativeOfFunction(LayerOutputs);
			outputGradients.Multiply(errors);
			outputGradients.ScalarMultiply(learningRate);
			Gradients.Add(outputGradients);
		}

		public void ApplyGradients(NeuralNetMatrix previousLayer, int batchSize)
		{
			Gradients.ScalarDivide(batchSize);

			//Calculate the output deltas
			var previousLayerTransposition = previousLayer.Transpose();
			var outputDeltas = NeuralNetMatrix.DotProduct(Gradients, previousLayerTransposition);

			//Update the Weights and Biases
			Weights.Add(outputDeltas);
			Biases.Add(Gradients);

			Gradients = new NeuralNetMatrix(Gradients.RowLength, Gradients.ColoumnLength);
		}
	}
}
