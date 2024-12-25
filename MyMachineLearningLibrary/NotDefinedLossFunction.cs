using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class NotDefinedLossFunction : ILossFunction
	{
		public NeuralNetMatrix CalculateDerivativeOfLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			throw new NotImplementedException();
		}

		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs)
		{
			throw new NotImplementedException();
		}
	}
}
