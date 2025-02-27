using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class BinaryCrossEntropyLossFunction : ILossFunction
	{
		public NeuralNetMatrix CalculateDerivativeOfLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var lossMatrix = outputs.Subtract(targets);
			var oneMinusOutputs = NeuralNetMatrix.Subtract(1, outputs);
			var x = outputs.Multiply(oneMinusOutputs);
			lossMatrix = lossMatrix.Divide(x);
			return lossMatrix;
		}

		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var lossMatrix = targets.Copy();
			var logOutputs = outputs.Log();
			lossMatrix = lossMatrix.Multiply(logOutputs);

			var oneMinusTargets = NeuralNetMatrix.Subtract(1, targets);
			var oneMinusOutputs = NeuralNetMatrix.Subtract(1, outputs);
			var logOneMinusOutputs = oneMinusOutputs.Log();
			var y = oneMinusTargets.Multiply(logOneMinusOutputs);

			lossMatrix = lossMatrix.Add(y);

			lossMatrix = lossMatrix.Multiply(-1);

			return lossMatrix.Average;
		}
	}
}
