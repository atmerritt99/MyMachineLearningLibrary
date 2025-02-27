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
			var lossMatrix = targets.Subtract(outputs);
			lossMatrix = lossMatrix.Multiply(lossMatrix);

			return lossMatrix.Average;
		}

		public NeuralNetMatrix CalculateDerivativeOfLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var derivativeOfLossMatrix = targets.Subtract(outputs);
			derivativeOfLossMatrix = derivativeOfLossMatrix.Multiply(-2);
			derivativeOfLossMatrix = derivativeOfLossMatrix.Divide(targets.RowLength);
			return derivativeOfLossMatrix;
		}
	}
}
