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
			//Assumes Developer uses Softmax activation function
			return outputs.Subtract(targets);
		}

		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var targetsCopy = targets.Copy();
			targetsCopy = targetsCopy.Multiply(outputs.Log());
			return targetsCopy.Sum * -1;
		}
	}
}
