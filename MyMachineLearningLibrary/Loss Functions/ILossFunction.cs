using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Loss_Functions
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<ILossFunction>))]
	public interface ILossFunction
	{
		public double CalculateLoss(MatrixExtension targets, MatrixExtension outputs);
		public MatrixExtension CalculateDerivativeOfLoss(MatrixExtension targets, MatrixExtension outputs);
	}
}
