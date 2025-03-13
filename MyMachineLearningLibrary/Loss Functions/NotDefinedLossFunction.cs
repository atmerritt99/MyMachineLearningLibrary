using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Loss_Functions
{
	public class NotDefinedLossFunction : ILossFunction
	{
		public MatrixExtension CalculateDerivativeOfLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			throw new NotImplementedException();
		}

		public double CalculateLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			throw new NotImplementedException();
		}
	}
}
