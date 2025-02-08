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
			var lossMatrix = NeuralNetMatrix.Subtract(outputs, targets);
			var oneMinusOutputs = NeuralNetMatrix.ScalarSubtract(1, outputs);
			var x = NeuralNetMatrix.Multiply(outputs, oneMinusOutputs);
			lossMatrix.Divide(x);
			return lossMatrix;
		}

		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			var lossMatrix = targets.Copy();
			var logOutputs = NeuralNetMatrix.Log(outputs);
			lossMatrix.Multiply(logOutputs);

			var oneMinusTargets = NeuralNetMatrix.ScalarSubtract(1, targets);
			var oneMinusOutputs = NeuralNetMatrix.ScalarSubtract(1, outputs);
			var logOneMinusOutputs = NeuralNetMatrix.Log(oneMinusOutputs);
			var y = NeuralNetMatrix.Multiply(oneMinusTargets, logOneMinusOutputs);

			lossMatrix.Add(y);

			lossMatrix.ScalarMultiply(-1);

			return lossMatrix.Average;
		}
	}
}
