using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Loss_Functions
{
	public class MeanSquaredErrorLossFunction : ILossFunction
	{
		public double CalculateLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			var lossMatrix = targets.Subtract(outputs);
			lossMatrix = lossMatrix.Multiply(lossMatrix);

			return lossMatrix.Average;
		}

		public MatrixExtension CalculateDerivativeOfLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			var derivativeOfLossMatrix = targets.Subtract(outputs);
			derivativeOfLossMatrix = derivativeOfLossMatrix.Multiply(-2);
			return derivativeOfLossMatrix;
		}
	}
}
