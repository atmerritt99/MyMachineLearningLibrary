using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<ILossFunction>))]
	public interface ILossFunction
	{
		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs);
		public NeuralNetMatrix CalculateDerivativeOfLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs);
	}
}
