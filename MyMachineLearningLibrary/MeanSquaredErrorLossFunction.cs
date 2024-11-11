using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class MeanSquaredErrorLossFunction : ILossFunction
	{
		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var lossMatrix = NeuralNetMatrix.Subtract(targets, outputs);
			lossMatrix.Multiply(lossMatrix);

			return lossMatrix.Average;
		}

		public NeuralNetMatrix CalculateDerivativeOfLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var derivativeOfLossMatrix = NeuralNetMatrix.Subtract(targets, outputs);
			derivativeOfLossMatrix.ScalarMultiply(-2);
			derivativeOfLossMatrix.ScalarDivide(targets.RowLength);
			return derivativeOfLossMatrix;
		}
	}
}
