using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Weight_Initialization
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<IWeightInitializtion>))]
	public interface IWeightInitializtion
	{
		public void InitializeWeights(NeuralNetwork neuralNetwork);
	}
}
