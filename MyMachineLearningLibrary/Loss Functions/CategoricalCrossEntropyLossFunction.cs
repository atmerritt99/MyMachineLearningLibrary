using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Loss_Functions
{
	public class CategoricalCrossEntropyLossFunction : ILossFunction
	{
		public MatrixExtension CalculateDerivativeOfLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			//Assumes Developer uses Softmax activation function
			return outputs.Subtract(targets);
		}

		public double CalculateLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			var targetsCopy = targets.Copy();
			targetsCopy = targetsCopy.Multiply(outputs.Log());
			return targetsCopy.Sum * -1 / targets.RowLength;
		}
	}
}
