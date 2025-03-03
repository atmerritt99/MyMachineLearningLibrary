using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class DenseLayer : ILayer
	{
		public int LayerIndex { get; set; }
		public double Alpha { get ; set; } // Learning Rate
		public double Beta1 { get; set; }
		public double Beta2 { get; set; }
		public int DecayRate { get; set; }
		public NeuralNetMatrix M { get; set; } // First Momentum
		public NeuralNetMatrix V { get; set; } // Second Momentum
		public double Epsilon { get; set; }
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
			Alpha = .01;
			Beta1 = .9;
			Beta2 = .999;
			Epsilon = 1e-8;
			DecayRate = 1;

			NumberOfPerceptrons = numberOfPerceptrons;
			Perceptrons = new DensePerceptron[numberOfPerceptrons];
			ActivationFunction = activationFunction;
		}

		public NeuralNetMatrix Backpropagate(NeuralNetMatrix errors, double learningRate, double decayRate, NeuralNetMatrix previousLayer, int batchSize, int currentEpoch, IOptimizer optimizer)
		{
			//Derivative of Softmax isn't used since that is implemented within the Categorical Cross Entropy Loss Function
			Gradients = typeof(SoftmaxActivationFunction) == ActivationFunction.GetType() ? LayerOutputs : ActivationFunction.ActivateDerivativeOfFunction(LayerOutputs);

			Gradients = Gradients.Multiply(errors);
			Gradients = Gradients.Divide(batchSize);

			var x = optimizer.OptimizeGradients(Gradients, learningRate, decayRate, currentEpoch, LayerIndex);

			var previousLayerTransposition = previousLayer.Transpose();
			var outputDeltas = x.DotProduct(previousLayerTransposition);

			//Update the Weights and Biases With Gradient Descent Optimization
			for (int i = 0; i < Perceptrons.Length; i++)
			{
				for (int j = 0; j < Perceptrons[i].Weights.Length; j++)
				{
					Perceptrons[i].Weights[j] -= outputDeltas[i, j];
				}
			}

			for (int i = 0; i < Perceptrons.Length; i++)
			{
				Perceptrons[i].Bias -= x[i, 0];
			}

			// Reset the Gradients after applying them
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

		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, ILayer previousLayer, int layerIndex)
		{
			LayerIndex = layerIndex;

			for(int i = 0; i < Perceptrons.Length; i++)
			{
				Perceptrons[i] = new DensePerceptron(numberOfPerceptronsInPreviousLayer, ActivationFunction);
			}

			Gradients = new NeuralNetMatrix(Perceptrons.Length, 1);
			M = new NeuralNetMatrix(Gradients.RowLength, Gradients.ColoumnLength);
			V = new NeuralNetMatrix(Gradients.RowLength, Gradients.ColoumnLength);
		}

		public NeuralNetMatrix TransposeWeights()
		{
			return Weights.Transpose();
		}
	}
}
