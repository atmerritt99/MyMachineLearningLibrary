using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Loss_Functions
{
	public class MeanAbsoluteErrorLossFunction : ILossFunction
	{
		public MatrixExtension CalculateDerivativeOfLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			return MatrixExtension.Compare(outputs, targets);
		}

		public double CalculateLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			var lossMatrix = targets.Subtract(outputs);
			lossMatrix = lossMatrix.AbsoluteValue();

			return lossMatrix.RowAverage;
		}
	}
}
