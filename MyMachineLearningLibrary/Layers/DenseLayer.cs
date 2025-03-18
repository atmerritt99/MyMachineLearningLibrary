using MyMachineLearningLibrary.Activation_Functions;
using MyMachineLearningLibrary.Optimizers;
using MyMachineLearningLibrary.Perceptrons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Layers
{
	public class DenseLayer : ILayer
	{
		public int LayerIndex { get; set; }
		public int NumberOfPerceptrons { get; set; }
		public IActivationFunction ActivationFunction { get; set; }
		public IPerceptron[] Perceptrons { get; set; }
		public MatrixExtension Gradients { get; set; }
		public MatrixExtension Weights
		{
			get
			{
				var weightsMatrix = new MatrixExtension(Perceptrons.Length, Perceptrons[0].Weights.Length);
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
		public MatrixExtension LayerOutputs
		{
			get
			{
				return ActivationFunction.ActivateFunction(new MatrixExtension(Perceptrons.Select(x => x.Output).ToArray()));
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
			Gradients = new MatrixExtension();
		}

		public MatrixExtension Backpropagate(MatrixExtension errors, double learningRate, double decayRate, MatrixExtension previousLayerOutputs, int batchSize, int currentEpoch, IOptimizer optimizer)
		{
			//Derivative of Softmax isn't used since that is implemented within the Categorical Cross Entropy Loss Function
			Gradients = typeof(SoftmaxActivationFunction) == ActivationFunction.GetType() ? LayerOutputs : ActivationFunction.ActivateDerivativeOfFunction(LayerOutputs);

			Gradients = Gradients.Multiply(errors);
			//Gradients = Gradients.Divide(batchSize); // Commented out until batch descent is fixed

			var x = optimizer.OptimizeGradients(Gradients, learningRate, decayRate, currentEpoch, LayerIndex);

			var previousLayerTransposition = previousLayerOutputs.Transpose();
			var weightDeltas = x.DotProduct(previousLayerTransposition);

			//Update the Weights and Biases With Gradient Descent Optimization
			for (int i = 0; i < Perceptrons.Length; i++)
			{
				for (int j = 0; j < Perceptrons[i].Weights.Length; j++)
				{
					Perceptrons[i].Weights[j] -= weightDeltas[i, j];
				}
			}

			for (int i = 0; i < Perceptrons.Length; i++)
			{
				//Bias deltas are the gradients passed through the optimizer
				Perceptrons[i].Bias -= x[i, 0];
			}

			// Reset the Gradients after applying them
			Gradients = new MatrixExtension(Gradients.RowLength, Gradients.ColoumnLength);

			return TransposeWeights().DotProduct(errors);
		}

		public MatrixExtension FeedForward(MatrixExtension layerInputs)
		{
			for (int i = 0; i < Perceptrons.Length; i++)
			{
				Perceptrons[i].WeightedSum(layerInputs.Flatten());
			}
			return LayerOutputs;
		}

		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, int layerIndex)
		{
			LayerIndex = layerIndex;

			for (int i = 0; i < Perceptrons.Length; i++)
			{
				Perceptrons[i] = new DensePerceptron(numberOfPerceptronsInPreviousLayer, ActivationFunction);
			}

			Gradients = new MatrixExtension(Perceptrons.Length, 1);
		}

		public MatrixExtension TransposeWeights()
		{
			return Weights.Transpose();
		}
	}
}
