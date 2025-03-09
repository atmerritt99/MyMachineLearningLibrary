using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<IPerceptron>))]
	public interface IPerceptron
	{
		public double Output { get; set; }
		public double[] Weights { get; set; }
		public double Bias { get; set; }
		public void NormalRandomizeWeights(double mean = 0, double standardDeviation = .1);
		public void UniformRandomizeWeights(double min = -1, double max = 1);
		public double Activate(double[] inputs);
		public double WeightedSum(double[] inputs);
	}
}
