using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Optimizers
{
	public class AdamOptimizer : IOptimizer
	{
		public MatrixExtension[] M { get; set; }
		public MatrixExtension[] V { get; set; }
		public double Beta1 { get; set; } = .9;
		public double Beta2 { get; set; } = .999;
		public double Epsilon { get; set; } = 1e-8;
		public AdamOptimizer()
		{
			M = [];
			V = [];
		}
		public AdamOptimizer(double beta1, double beta2, double epsilon)
		{
			Beta1 = beta1;
			Beta2 = beta2;
			Epsilon = epsilon;
			M = [];
			V = [];
		}
		public void Compile(NeuralNetwork neuralNetwork)
		{
			M = new MatrixExtension[neuralNetwork.Layers.Count - 1];
			V = new MatrixExtension[neuralNetwork.Layers.Count - 1];

			for (int i = 1; i < neuralNetwork.Layers.Count; i++) // Do not include the input layer
			{
				M[i - 1] = new MatrixExtension(neuralNetwork.Layers[i].Gradients.RowLength, neuralNetwork.Layers[i].Gradients.ColoumnLength);
				V[i - 1] = new MatrixExtension(neuralNetwork.Layers[i].Gradients.RowLength, neuralNetwork.Layers[i].Gradients.ColoumnLength);
			}
		}
		public MatrixExtension OptimizeGradients(MatrixExtension gradients, double learningRate, double decayRate, int currentEpoch, int layerIndex)
		{
			int gradientsIndex = layerIndex - 1; // Since Input layer is not included

			var t = gradients.Multiply(1 - Beta1);
			M[gradientsIndex] = M[gradientsIndex].Multiply(Beta1).Add(t);

			//Update Second Momentum Weights
			var g2 = gradients.Multiply(gradients);
			t = g2.Multiply(1 - Beta2);
			V[gradientsIndex] = V[gradientsIndex].Multiply(Beta2).Add(t);

			//Calculate Decay
			var decay = currentEpoch / decayRate;

			//Remove 0 Bias
			var mhat = M[gradientsIndex].Divide(1 - Math.Pow(Beta1, currentEpoch) + Epsilon);
			var vhat = V[gradientsIndex].Divide(1 - Math.Pow(Beta2, currentEpoch) + Epsilon);

			var alpha = decayRate == 0 ? learningRate : learningRate / (Math.Sqrt(decay) + Epsilon);

			t = vhat.SquareRoot().Add(Epsilon);
			return mhat.Multiply(alpha).Divide(t);
		}
	}
}
