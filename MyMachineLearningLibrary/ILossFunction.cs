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
		public NeuralNetMatrix CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs, out double averageLoss);
	}
}
