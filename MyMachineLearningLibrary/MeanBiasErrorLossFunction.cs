using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class MeanBiasErrorLossFunction : ILossFunction
	{
		public NeuralNetMatrix CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs, out double averageLoss)
		{
			var lossMatrix = NeuralNetMatrix.Subtract(targets, outputs);

			averageLoss = lossMatrix.Average;

			return lossMatrix;
		}
	}
}
