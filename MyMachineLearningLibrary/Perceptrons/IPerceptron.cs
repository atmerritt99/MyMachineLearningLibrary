using MyMachineLearningLibrary.Activation_Functions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Perceptrons
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<IPerceptron>))]
	public interface IPerceptron
	{
		public double Output { get; set; }
		public double[] Weights { get; set; }
		public double Bias { get; set; }
		public IActivationFunction ActivationFunction { get; set; }
		public void NormalRandomizeWeights(double mean = 0, double standardDeviation = .1);
		public void UniformRandomizeWeights(double min = -1, double max = 1);
		public void NormalRandomizeBias(double mean = 0, double standardDeviation = .1);
		public void UniformRandomizeBias(double min = -1, double max = 1);
		public double Activate(double[] inputs);
		public double WeightedSum(double[] inputs);
	}
}
