using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class MeanSquaredErrorLossFunction : ILossFunction
	{
		public NeuralNetMatrix CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs, out double averageLoss)
		{
			var targetsSquared = NeuralNetMatrix.Multiply(targets, targets);
			var outputsSquared = NeuralNetMatrix.Multiply(outputs, outputs);
			var lossMatrix = NeuralNetMatrix.Subtract(targetsSquared, outputsSquared);

			averageLoss = lossMatrix.Average;

			return lossMatrix;
		}
	}
}
