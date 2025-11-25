using MyMachineLearningLibrary.Activation_Functions;
using MyMachineLearningLibrary.Optimizers;
using MyMachineLearningLibrary.Perceptrons;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
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

		public MatrixExtension Biases
		{
			get
			{
				return new MatrixExtension(Perceptrons.Select(x => x.Bias).ToArray());
			}
			set
			{

			}
		}

		public MatrixExtension Outputs { get; set; }

		public DenseLayer(int numberOfPerceptrons, IActivationFunction activationFunction)
		{
			NumberOfPerceptrons = numberOfPerceptrons;
			Perceptrons = new DensePerceptron[numberOfPerceptrons];
			ActivationFunction = activationFunction;
			Gradients = new MatrixExtension();
			Outputs = new MatrixExtension();
		}

		public MatrixExtension Backpropagate(MatrixExtension errors, double learningRate, double decayRate, MatrixExtension previousLayerOutputs, int currentEpoch, IOptimizer optimizer)
		{
			//Derivative of Softmax isn't used since that is implemented within the Categorical Cross Entropy Loss Function
			Gradients = typeof(SoftmaxActivationFunction) == ActivationFunction.GetType() ? Outputs : ActivationFunction.ActivateDerivativeOfFunction(Outputs);

			Gradients = Gradients.Multiply(errors);

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
				//Bias deltas are the gradients passed through the optimizer
				Perceptrons[i].Bias -= x[i, 0];
			}

			return TransposeWeights().DotProduct(errors);
		}

		public MatrixExtension FeedForward(MatrixExtension layerInputs)
		{
			Outputs = ActivationFunction.ActivateFunction(Weights.DotProduct(layerInputs).Add(Biases));
			return Outputs;
		}

		public void InitializeLayer(int numberOfPerceptronsInPreviousLayer, int layerIndex)
		{
			LayerIndex = layerIndex;

			for (int i = 0; i < Perceptrons.Length; i++)
			{
				Perceptrons[i] = new DensePerceptron(numberOfPerceptronsInPreviousLayer, ActivationFunction);
			}

			// Is here so Network can compile correctly
			// See: Optimizer Compile Function
			Gradients = new MatrixExtension(Perceptrons.Length, 1);
		}

		public MatrixExtension TransposeWeights()
		{
			return Weights.Transpose();
		}
	}
}
