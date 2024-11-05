using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class BinaryCrossEntropyLossFunction : ILossFunction
	{
		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs, out NeuralNetMatrix lossMatrix, out NeuralNetMatrix errorsDirections)
		{
			lossMatrix = targets.Copy();
			var logOutputs = NeuralNetMatrix.NaturalLog(outputs);
			lossMatrix.Multiply(logOutputs);

			var oneMinusTargets = NeuralNetMatrix.ScalarSubtract(1, targets);
			var oneMinusOutputs = NeuralNetMatrix.ScalarSubtract(1, outputs);
			var logOneMinusOutputs = NeuralNetMatrix.NaturalLog(oneMinusOutputs);
			var y = NeuralNetMatrix.Multiply(oneMinusTargets, logOneMinusOutputs);

			lossMatrix.Add(y);

			lossMatrix.ScalarMultiply(-1);

			errorsDirections = NeuralNetMatrix.Compare(targets, outputs);

			return lossMatrix.Average;
		}
	}
}
