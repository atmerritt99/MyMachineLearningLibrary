using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class CategoricalCrossEntropyLossFunction : ILossFunction
	{
		public NeuralNetMatrix CalculateDerivativeOfLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			return NeuralNetMatrix.Subtract(outputs, targets);
		}

		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var targetsCopy = targets.Copy();
			targetsCopy.Multiply(NeuralNetMatrix.Log(outputs));
			return targetsCopy.Sum * -1;
		}
	}
}
