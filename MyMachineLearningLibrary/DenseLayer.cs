using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class DenseLayer : ILayer
	{
		public int NumberOfPerceptrons { get; set; }
		public IActivationFunction ActivationFunction { get; set; }
		public IPerceptron[] Perceptrons { get; set; }
		public NeuralNetMatrix Gradients { get; set; }
		public NeuralNetMatrix Weights
		{ 
			get
			{
				var weightsMatrix = new NeuralNetMatrix(Perceptrons.Length, Perceptrons[0].Weights.Length);
				for (int i = 0; i < Perceptrons.Length; i++)
				{
					weightsMatrix.Values[i] = Perceptrons[i].Weights;
				}
				return weightsMatrix;
			}
			set
			{

			}
		}
		public NeuralNetMatrix LayerOutputs
		{
			get
			{
				return ActivationFunction.ActivateFunction(new NeuralNetMatrix(Perceptrons.Select(x => x.Output).ToArray()));
			}
			set
			{

			}
		}

		public DenseLayer(int numberOfPerceptrons, IActivationFunction activationFunction)
		{
			NumberOfPerceptrons = numberOfPerceptrons;
			Perceptrons = new DensePerceptron[numberOfPerceptrons];
			ActivationFunction = activationFunction;
		}

		public NeuralNetMatrix Backpropagate(NeuralNetMatrix errors, double learningRate, NeuralNetMatrix previousLayer, int batchSize)
		{
			var outputGradients = typeof(SoftmaxActivationFunction) == ActivationFunction.GetType() ? LayerOutputs : ActivationFunction.ActivateDerivativeOfFunction(LayerOutputs);
			outputGradients = outputGradients.Multiply(errors);
			outputGradients = outputGradients.Multiply(learningRate);
			Gradients = Gradients.Add(outputGradients);

			Gradients = Gradients.Divide(batchSize);

			//Calculate the output deltas
			var previousLayerTransposition = previousLayer.Transpose();
			var outputDeltas = Gradients.DotProduct(previousLayerTransposition);

			//Update the Weights and Biases With Gradient Descent Optimization
			//Weights.Subtract(outputDeltas);
			for (int i = 0; i < Perceptrons.Length; i++)
			{
				for (int j = 0; j < Perceptrons[i].Weights.Length; j++)
				{
					Perceptrons[i].Weights[j] -= outputDeltas[i, j];
				}
			}

			//Biases.Subtract(Gradients);
			for (int i = 0; i < Perceptrons.Length; i++)
			{
				Perceptrons[i].Bias -= Gradients[i, 0];
			}

			Gradients = new NeuralNetMatrix(Gradients.RowLength, Gradients.ColoumnLength);

			return TransposeWeights().DotProduct(errors);
		}

		public NeuralNetMatrix FeedForward(NeuralNetMatrix layerInputs)
		{
			for(int i = 0; i < Perceptrons.Length; i++)
			{
				Perceptrons[i].WeightedSum(layerInputs.Flatten());
			}
			return LayerOutputs;
		}

		//public NeuralNetMatrix TransposeWeights()
		//{
		//	return Weights.Transpose();
		//}

		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, NeuralNetMatrix previousLayersGradients)
		{
			for(int i = 0; i < Perceptrons.Length; i++)
			{
				Perceptrons[i] = new DensePerceptron(numberOfPerceptronsInPreviousLayer, ActivationFunction);
			}

			Gradients = Weights.DotProduct(previousLayersGradients);
		}

		public NeuralNetMatrix TransposeWeights()
		{
			return Weights.Transpose();
		}
	}
}
