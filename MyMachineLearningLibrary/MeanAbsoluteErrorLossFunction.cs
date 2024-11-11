using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class MeanAbsoluteErrorLossFunction : ILossFunction
	{
		public NeuralNetMatrix CalculateDerivativeOfLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			return NeuralNetMatrix.Compare(outputs, targets);
		}

		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var lossMatrix = NeuralNetMatrix.Subtract(targets, outputs);
			lossMatrix.ScalarAbs();

			return lossMatrix.Average;
		}
	}
}
